"""Makes a model answer the item banks: the probability of each option.

Usage:  python3 gpu/machines_llm.py HF_ID NAME TASKS MODES [--output gpu-results]

MODES is a list among `letter` (immediate answer) and `reason` (reasoning first,
then the constrained letter). The choice is always constrained to the valid letters and the
probability of each is read, so the machine gives a verdict and a confidence.
Resumes where it stopped: an item already written is not redone.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
LETTERS = "ABCDEFGHIJKLMNOP"


def format_prompt(c: dict, mode: str) -> str:
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(c["options"]))
    last = LETTERS[len(c["options"]) - 1]
    if mode == "letter":
        tail = f"Reply with only the letter of the correct option (A to {last})."
    else:
        tail = f"Think step by step, briefly. Then finish with a final line of the form: Answer: X, where X is a letter from A to {last}."
    return f"{c['text']}\n\n{options}\n\n{tail}"


class Machine:
    def __init__(self, hf_id: str, images: bool):
        from vllm import LLM
        self.images = images
        self.llm = LLM(model=hf_id, max_model_len=int(os.environ.get("MAX_LEN", 4096)), max_num_seqs=16,
                       gpu_memory_utilization=float(os.environ.get("MEM_GPU", 0.90)),
                       enable_prefix_caching=True, enforce_eager=True, trust_remote_code=False,
                       limit_mm_per_prompt={"image": 1 if images else 0, "video": 0}, seed=534)
        self.tok = self.llm.get_tokenizer()
        self.processor = None
        if images:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(hf_id)

    def prompt(self, text: str, with_image: bool) -> str:
        cont = [{"type": "image"}, {"type": "text", "text": text}] if with_image else text
        who = self.processor if with_image else self.tok
        return who.apply_chat_template([{"role": "user", "content": cont}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def ids(self, k: int, space: bool) -> list[int]:
        out = []
        for l in LETTERS[:k]:
            t = self.tok.encode((" " if space else "") + l, add_special_tokens=False)
            if len(t) != 1:
                t = self.tok.encode(l, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"letter {l} is not a single token: {t}")
            out.append(t[0])
        return out

    def letter(self, inputs: list, ids_k: list[list[int]]) -> list[list[float]]:
        """A distribution over the allowed letters for each input."""
        from vllm import SamplingParams
        import math
        params = [SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=i, logprobs=len(i)) for i in ids_k]
        outputs = self.llm.generate(inputs, params, use_tqdm=False)
        everything = []
        for u, ids in zip(outputs, ids_k):
            d = u.outputs[0].logprobs[0]
            lp = [d[i].logprob if i in d else -40.0 for i in ids]
            m = max(lp)
            weights = [math.exp(x - m) for x in lp]
            s = sum(weights)
            everything.append([p / s for p in weights])
        return everything


def load_image(base: Path, c: dict):
    from PIL import Image
    im = Image.open(base / c["image"]).convert("RGB")
    if os.environ.get("SIDE_MAX"):            # large photos do not fit in the context of the small vision models
        im.thumbnail((int(os.environ["SIDE_MAX"]),) * 2)
    return im


def model_input(m: Machine, base: Path, c: dict, text: str):
    if c.get("image"):
        return {"prompt": text, "multi_modal_data": {"image": load_image(base, c)}}
    return {"prompt": text}


def main() -> None:
    from vllm import SamplingParams
    hf_id, name, tasks, modes = sys.argv[1:5]
    output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path(os.environ["RND"]) / "gpu-results"
    base = Path("data")
    output.mkdir(parents=True, exist_ok=True)
    everything = {t: [json.loads(l) for l in open(base / "items" / f"{t}.jsonl", encoding="utf-8")] for t in tasks.split(",")}
    images = any(c.get("image") for cs in everything.values() for c in cs)
    t0 = time.time()
    machine = Machine(hf_id, images)
    print(f"loaded {name} in {time.time() - t0:.0f} s", flush=True)
    for task, items in everything.items():
        for mode in modes.split(","):
            file = output / f"{task}__{name}__{mode}.jsonl"
            done = {json.loads(l)["id"] for l in open(file, encoding="utf-8")} if file.exists() else set()
            rest = [c for c in items if c["id"] not in done]
            t = time.time()
            with open(file, "a", encoding="utf-8") as f:
                for i in range(0, len(rest), 64):
                    block = rest[i:i + 64]
                    texts = [machine.prompt(format_prompt(c, mode), bool(c.get("image"))) for c in block]
                    reasoning = [""] * len(block)
                    if mode == "reason":
                        p = SamplingParams(max_tokens=700, temperature=0.0, stop=["Answer:", "ANSWER:"])
                        u = machine.llm.generate([model_input(machine, base, c, x) for c, x in zip(block, texts)], p, use_tqdm=False)
                        reasoning = [o.outputs[0].text for o in u]
                        texts = [x + r + "Answer:" for x, r in zip(texts, reasoning)]
                    ids = [machine.ids(len(c["options"]), space=mode == "reason") for c in block]
                    prob = machine.letter([model_input(machine, base, c, x) for c, x in zip(block, texts)], ids)
                    for c, p, r in zip(block, prob, reasoning):
                        f.write(json.dumps({"id": c["id"], "prob": [round(x, 6) for x in p], "correct_option": c["correct_option"],
                                            "length": len(r), "head": r[:160]}, ensure_ascii=False) + "\n")
                    f.flush()
            correct = n = 0
            for l in open(file, encoding="utf-8"):
                d = json.loads(l)
                n += 1
                correct += max(range(len(d["prob"])), key=d["prob"].__getitem__) == d["correct_option"]
            print(f"{task} {mode}: {n} items, accuracy {correct / n:.3f}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
