"""ChatBench with the authors' machines: AI alone, AI in chat, who decides first."""
import json
from pathlib import Path

import numpy as np

from . import sources, sdt
from .measure import compare

def convert_person_person(row: dict, acc: float, m: int = 4) -> dict:
    """ρ between two people, from the correct answers to the cues, as for person and machine.

    With four options the correlation of the correct answers is not that of the cues. Here
    the conversion assumes two people with the group's mean sensitivity: it is an
    approximation, because it does not simulate the personal thresholds as the
    person-machine one does.
    """
    dh = sdt.d_from_accuracy(acc, m)
    conv = [sdt.convert_rho_mafc(float(r), dh, dh, m) for r in (row["rho_hh"], *row["ci"])]
    return {"dh_group": dh, "rho_hh_cues": conv[0], "ci_cues": conv[1:], "phi_cues": sdt.angle_degrees(conv[0]),
            "phi_cues_ci": sorted(sdt.angle_degrees(r) for r in conv[1:])}


base = Path("data")
c = sources.chatbench_pairs(base)
res, table = {}, []
in_chat = sources.chatbench_machine_in_chat(base)
for model in ("gpt-4o", "llama-3.1-8b"):
    m = sources.chatbench_machine_alone(base, model)
    x = c[c.model == model]
    for subj in (None, "math", "physics", "moral"):
        for kind, p in (("alone", sources.chatbench_trials(x, m, subj)), ("in-chat", sources.chatbench_trials_in_chat(c, in_chat, model, subj))):
            r = compare(p, m=4, repetitions=2000)
            r.update(machine=model, subject=subj or "all", kind=kind)
            table.append(r)
            print(model, subj, kind, {k: round(r[k], 3) for k in ("acc_h", "acc_m", "acc_s", "phi", "useful_angle", "k", "g_90", "g_phi", "eta_phi") if r.get(k) is not None}, r.get("phi_ci", ""), flush=True)
res["comparisons"] = table

# The same pair at the two terminals: the machine alone, on the conversations that have an answer in chat only.
same = []
for model in ("gpt-4o", "llama-3.1-8b"):
    m = sources.chatbench_machine_alone(base, model)
    keys = in_chat[in_chat.model == model].dropna(subset=["m_ok"])[["worker_id", "question_id"]].drop_duplicates()
    x = c[c.model == model].merge(keys, on=["worker_id", "question_id"])
    r = compare(sources.chatbench_trials(x, m, None), m=4, repetitions=2000)
    r.update(machine=model, subject="all", kind="alone-same-trials")
    same.append(r)
    print(model, "alone, same trials", {k: round(r[k], 3) for k in ("acc_h", "acc_m", "acc_s", "phi", "g_phi", "eta_phi", "turing") if r.get(k) is not None}, r.get("phi_ci", ""), flush=True)
res["same_trials"] = same

# between people: ρ of the second person on the same question
hh = []
for subj in (None, "math", "physics", "moral"):
    d = c if subj is None else c[c.subject == subj]
    from .sources import _index
    p, _ = _index(d.worker_id); q, _ = _index(d.question_id)
    r0 = sdt.rho_person_pairs(p, q, d.acc.to_numpy(float))
    gen = np.random.default_rng(534)
    boot = []
    Q = q.max() + 1; P = p.max() + 1
    for _ in range(300):
        wq = gen.poisson(1.0, Q).astype(float)
        boot.append(sdt.rho_person_pairs(p, q, d.acc.to_numpy(float), item_weights=wq))
    row = {"subject": subj or "all", "rho_hh": r0, "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "phi": sdt.angle_degrees(r0)}
    row.update(convert_person_person(row, float(d.acc.mean())))
    hh.append(row)
    print("person-person", hh[-1], flush=True)
res["person_person"] = hh

# H6: who decides first
u = sources.chatbench_all_answers(base)
u = u[u.question_id.str.contains("-redux-") & (u.answer_type == "user-AI") & u.condition.isin(["answer-first", "direct-to-AI"])].dropna(subset=["acc"])
gen = np.random.default_rng(534)
rows = []
def _boot(d, gen):
    ww = d.worker_id.unique()
    weights = dict(zip(ww, gen.poisson(1.0, ww.size)))
    w = d.worker_id.map(weights).to_numpy(float)
    return np.sum(w * d.acc.to_numpy(float)) / np.sum(w)
for model in ("gpt-4o", "llama-3.1-8b"):
    for subj in (None, "math", "physics", "moral"):
        d = u[u.model == model] if subj is None else u[(u.model == model) & (u.subject == subj)]
        af, da = d[d.condition == "answer-first"], d[d.condition == "direct-to-AI"]
        delta = af.acc.mean() - da.acc.mean()
        boot = [_boot(af, gen) - _boot(da, gen) for _ in range(2000)]
        r = {"machine": model, "subject": subj or "all", "acc_answer_first": float(af.acc.mean()), "acc_direct": float(da.acc.mean()),
             "n_af": int(len(af)), "n_da": int(len(da)), "delta": float(delta), "delta_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        rows.append(r); print("H6", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
res["who_decides_first"] = rows
Path("results").mkdir(exist_ok=True)
Path("results/chatbench-source.json").write_text(json.dumps(res, indent=1))
