"""An open model in the three conditions of the devil's advocate, with vLLM.

Usage, from the experiment folder, on the GPU (after gpu/environment.sh and the venv):

  python3 gpu/devil_llm.py HF_ID NAME [--output DIR]

Same machinery as gpu/machines_llm.py: chat template, greedy decoding, the letter read from the probabilities
constrained to the valid options. `alone` asks the questions of data/items/chatbench.jsonl with the `letter`
prompt; `told` and `devil` ask data/items/chatbench-devil.jsonl with angle.devil.prompt. In `devil` the arguments
are generated first (up to 700 tokens, stopped at «Answer:»), then the letter is read after «Answer:», as in the
`reason` mode of the other open models.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angle.devil_prompt import prompt                                    # noqa: E402
from gpu.machines_llm import Machine, model_input                 # noqa: E402

BASE = Path("data")
RUNS = (("alone", "chatbench"), ("told", "chatbench-devil"), ("devil", "chatbench-devil"))
# controls added after Llama's results (PROTOCOL.md): the same reasoning as `devil`, without and with the answer
CONTROLS = (("alone_reason", "chatbench"), ("told_reason", "chatbench-devil"))


def main() -> None:
    from vllm import SamplingParams
    hf_id, name = sys.argv[1:3]
    output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path("devil-results")
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    machine = Machine(hf_id, False)
    print(f"loaded {name} in {time.time() - t0:.0f} s", flush=True)
    for mode, task in (CONTROLS if "--controls" in sys.argv else RUNS):
        items = [json.loads(l) for l in open(BASE / "items" / f"{task}.jsonl", encoding="utf-8")]
        file = output / f"{task}__{name}__{mode}.jsonl"
        done = {json.loads(l)["id"] for l in open(file, encoding="utf-8")} if file.exists() else set()
        rest = [c for c in items if c["id"] not in done]
        t = time.time()
        with open(file, "a", encoding="utf-8") as f:
            for i in range(0, len(rest), 64):
                block = rest[i:i + 64]
                texts = [machine.prompt(prompt(c, mode), False) for c in block]
                arguments = [""] * len(block)
                if mode in ("devil", "alone_reason", "told_reason"):
                    p = SamplingParams(max_tokens=700, temperature=0.0, stop=["Answer:", "ANSWER:"])
                    u = machine.llm.generate([model_input(machine, BASE, c, x) for c, x in zip(block, texts)], p, use_tqdm=False)
                    arguments = [o.outputs[0].text for o in u]
                    texts = [x + a + "Answer:" for x, a in zip(texts, arguments)]
                ids = [machine.ids(len(c["options"]), space=mode in ("devil", "alone_reason", "told_reason")) for c in block]
                prob = machine.letter([model_input(machine, BASE, c, x) for c, x in zip(block, texts)], ids)
                for c, pr, a in zip(block, prob, arguments):
                    f.write(json.dumps({"id": c["id"], "prob": [round(x, 6) for x in pr], "correct_option": c["correct_option"],
                                        "hypothesis": c.get("hypothesis"), "length": len(a), "head": a[:1000]},
                                       ensure_ascii=False) + "\n")
                f.flush()
        n = correct = 0
        for line in open(file, encoding="utf-8"):
            d = json.loads(line)
            n += 1
            correct += max(range(len(d["prob"])), key=d["prob"].__getitem__) == d["correct_option"]
        print(f"{task} {mode}: {n} items, accuracy {correct / max(n, 1):.3f}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    main()
