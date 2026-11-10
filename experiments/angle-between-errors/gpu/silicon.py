"""Silicon people: measure A of the assimilation protocol is tried on real networks.

Three arms, 20 networks each. A frozen machine M (ResNet) and its copy with a defect injected
on two classes. Each "person" is a small network that learns CIFAR-10 in stages; at the final exam, on
images never seen, for each network and at each stage it is recorded whether it answered correctly.

Usage, on the machine with the GPU:  python3 gpu/silicon.py OUTPUT
"""
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

DEV = "cuda"
SEED = 534
N_EXAM = 2000
PEOPLE = 20
STAGES = {"T0": 4000, "T1": 12000, "T3": 36000}      # images seen in total at each stage
EPOCHS = 6
ALPHA = 0.5                                          # how much the machine's opinion weighs


class Net(nn.Module):
    """The person: a small convolutional network."""
    def __init__(self):
        super().__init__()
        b = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.f = nn.Sequential(b(3, 48), b(48, 96), nn.MaxPool2d(2), b(96, 96), b(96, 192), nn.MaxPool2d(2), b(192, 192), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.c = nn.Linear(192, 10)

    def forward(self, x):
        return self.c(self.f(x))


class Block(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.a = nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o)
        self.b = nn.Conv2d(o, o, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o)
        self.s = nn.Identity() if s == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, stride=s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return F.relu(self.bn2(self.b(F.relu(self.bn1(self.a(x))))) + self.s(x))


class Machine(nn.Module):
    """The machine: a ResNet18 for CIFAR."""
    def __init__(self):
        super().__init__()
        self.ini = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        c = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        self.b = nn.Sequential(*[Block(*x) for x in c])
        self.fc = nn.Linear(512, 10)

    def features(self, x):
        return F.adaptive_avg_pool2d(self.b(self.ini(x)), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


def tensors(root):
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    conv = lambda x: (torch.tensor(x).permute(0, 3, 1, 2).float() / 255.0 - torch.tensor([.491, .482, .447])[None, :, None, None]) / torch.tensor([.247, .243, .262])[None, :, None, None]
    return conv(tr.data).to(DEV), torch.tensor(tr.targets).to(DEV), conv(te.data).to(DEV), torch.tensor(te.targets).to(DEV)


def augment(x):
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    b, _, h, w = pad.shape
    i = torch.randint(0, 9, (b,)); j = torch.randint(0, 9, (b,))
    out = torch.stack([pad[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32] for k in range(b)])
    flip = torch.rand(b, device=out.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def train(model, X, Y, epochs, lr=0.05, teacher=None, bs=128, gen=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = max(1, epochs * (len(X) // bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), generator=gen, device="cpu").to(DEV)
        for k in range(0, len(X) - bs + 1, bs):
            b = idx[k:k + bs]
            x = augment(X[b])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                log = model(x)
                loss = F.cross_entropy(log, Y[b])
                if teacher is not None:
                    with torch.no_grad():
                        t = F.softmax(teacher(x), dim=1)
                    loss = (1 - ALPHA) * loss + ALPHA * F.kl_div(F.log_softmax(log, dim=1), t, reduction="batchmean")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    return torch.cat([model(X[k:k + bs]).argmax(1) for k in range(0, len(X), bs)])


class Defective(nn.Module):
    """The injected defect: on the cats and dogs for which a fixed direction of the features exceeds a threshold,
    the machine swaps the two classes. It depends only on the image: whoever learns from it may inherit the pattern item by item.
    """
    def __init__(self, m, x_calibration, share=0.7):
        super().__init__(); self.m = m
        g = torch.Generator(device=DEV).manual_seed(SEED)
        self.register_buffer("u", F.normalize(torch.randn(512, generator=g, device=DEV), dim=0))
        with torch.no_grad():
            f = torch.cat([self.m.features(x_calibration[k:k + 512]) for k in range(0, len(x_calibration), 512)]).float()
        top = self.m.fc(f).argmax(1)
        z = f @ self.u
        self.register_buffer("threshold", torch.quantile(z[(top == 3) | (top == 5)], 1 - share))

    def forward(self, x):
        f = self.m.features(x)
        o = self.m.fc(f).clone()
        top = o.argmax(1)
        swap = (f.float() @ self.u) > self.threshold
        for a, b in ((3, 5), (5, 3)):
            sel = (top == a) & swap
            o[sel, b] = o[sel, a] + 2.0     # the opinion says the other class, with high confidence
        return o


def main() -> None:
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = tensors(str(output.parent / "cifar"))
    perm = np.random.default_rng(SEED).permutation(10000)
    exam, other = perm[:N_EXAM], perm[N_EXAM:]
    Xex, Yex = Xte[exam], Yte[exam]
    Xm = torch.cat([Xtr, Xte[other]]); Ym = torch.cat([Ytr, Yte[other]])
    M = Machine().to(DEV)
    train(M, Xm, Ym, 30, lr=0.1, gen=torch.Generator().manual_seed(SEED))
    pm = predict(M, Xex)
    print("machine exam accuracy", (pm == Yex).float().mean().item(), f"{time.time() - t0:.0f}s", flush=True)
    M.eval()
    for p in M.parameters():
        p.requires_grad_(False)
    D = Defective(M, Xex).to(DEV)
    pd = predict(D, Xex)
    print("defective exam accuracy", (pd == Yex).float().mean().item(), flush=True)
    np.savez_compressed(output / "machines.npz", exam=exam, y=Yex.cpu().numpy(), m=pm.cpu().numpy(), d=pd.cpu().numpy())
    arms = {"control": None, "assisted": M, "defect": D}
    for name, teacher in arms.items():
        for p in range(PEOPLE):
            s = SEED + 1000 * list(arms).index(name) + p
            torch.manual_seed(s); np.random.seed(s)
            gen = torch.Generator().manual_seed(s)
            order = torch.randperm(len(Xtr), generator=gen).to(DEV)
            net = Net().to(DEV)
            answers, done = {}, 0
            for stage, how_many in STAGES.items():
                new = order[done:how_many]
                seen = order[:how_many]
                train(net, Xtr[seen], Ytr[seen], EPOCHS if stage != "T0" else 6, lr=0.05,
                      teacher=None if stage == "T0" else teacher, gen=gen)
                answers[stage] = predict(net, Xex).cpu().numpy().astype(np.int16)
                done = how_many
            np.savez_compressed(output / f"{name}-{p:02d}.npz", **answers)
            acc = {k: float((torch.tensor(v).to(DEV) == Yex).float().mean()) for k, v in answers.items()}
            print(name, p, {k: round(v, 3) for k, v in acc.items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
