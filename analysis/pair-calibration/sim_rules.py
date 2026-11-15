#!/usr/bin/env python3
"""What each usage rule yields, inside the model, at the values measured in the article.

For each scenario, over 2 million cases simulated with the process of the repository:
  - the alarm: how often the two disagree, how often the machine is wrong where they agree
    and where they disagree, what share of its errors falls in a disagreement, who is right;
  - the rules: always follow the machine, always keep one's own answer, the ideal pair
    (graded evidence from both, optimally weighted), equal weights, the person who optimally
    weighs only the machine's verdict, and two threshold rules on the person's confidence
    ("I follow the machine, except when I am very confident"; "I keep mine, except when I am
    not very confident").

The person's confidence is here the distance between their first and second evidence: a
person who knows exactly how confident they are. Real people do not know this so well: these
numbers are a ceiling for the confidence rules, not a forecast.

Usage:  python3 sim_rules.py > results/rules.txt     # also writes results/rules.json and prints the tables
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import estimators as st
from sim_sample import SCENARI, SEME

N_TOT, BLOCCO = 2_000_000, 250_000
# English column labels for the printed tables (the dictionary keys stay as they are)
LABELS = {"persona": "person", "macchina": "machine", "coppia_ideale": "ideal_pair", "pesi_uguali": "equal_weights",
          "pesa_il_verdetto": "weighs_verdict", "macchina_tranne_molto_sicuro": "machine_unless_sure", "propria_tranne_poco_sicuro": "own_unless_unsure"}
BONUS = np.r_[np.arange(0.0, 4.01, 0.25), 5.0, 6.0, 8.0]


def scenario_binario(sc: dict) -> dict:
    """The same d' with only two options (forced choice between two)."""
    return dict(sc, m=2, nome=sc["nome"] + " [two options, same sensitivities]")


def corri(sc: dict, seed: int) -> dict:
    m, dh, dm, rho = sc["m"], sc["dh"], sc["dm"], sc["rho"]
    gen = np.random.default_rng(seed)
    den, num = dh - rho * dm, dm - rho * dh
    w_ideale = math.inf if den <= 1e-9 else (0.0 if num <= 0 else num / den)
    tot = dict(n=0, h=0, mok=0, dis=0, m_err_acc=0, m_err_dis=0, acc=0, dis_m=0, dis_h=0, ideale=0, uguali=0,
               molto=0, poco=0, dis_molto=0, dis_molto_h=0, dis_molto_m=0, dis_poco=0, dis_poco_h=0, dis_poco_m=0)
    verd = np.zeros(BONUS.size)
    segue = np.zeros(BONUS.size)
    # confidence thresholds: terciles of the distance between the person's first and second evidence
    ca, cb, a, b = st.scelte(BLOCCO, m, rho, dh, dm, gen)
    s = np.sort(a, axis=1)
    t_basso, t_alto = np.percentile(s[:, -1] - s[:, -2], [100 / 3, 200 / 3])
    for _ in range(N_TOT // BLOCCO):
        ca, cb, a, b = st.scelte(BLOCCO, m, rho, dh, dm, gen)
        n = ca.size
        righe = np.arange(n)
        ha, hb = ca == 0, cb == 0
        dis = ca != cb
        s = np.sort(a, axis=1)
        sic = s[:, -1] - s[:, -2]
        molto, poco = sic >= t_alto, sic < t_basso
        tot["n"] += n
        tot["h"] += ha.sum(); tot["mok"] += hb.sum(); tot["dis"] += dis.sum()
        tot["acc"] += (~dis).sum()
        tot["m_err_acc"] += (~hb & ~dis).sum(); tot["m_err_dis"] += (~hb & dis).sum()
        tot["dis_m"] += (dis & hb).sum(); tot["dis_h"] += (dis & ha).sum()
        if math.isinf(w_ideale):
            tot["ideale"] += hb.sum()
        else:
            tot["ideale"] += ((a + w_ideale * b).argmax(1) == 0).sum()
        tot["uguali"] += ((a + b).argmax(1) == 0).sum()
        # threshold rules on confidence
        tot["molto"] += np.where(molto, ha, hb).sum()        # follow the machine, except when very confident
        tot["poco"] += np.where(poco, hb, ha).sum()          # keep my own, except when not very confident
        for nome, sel in (("molto", molto), ("poco", poco)):
            tot["dis_" + nome] += (dis & sel).sum()
            tot["dis_" + nome + "_h"] += (dis & sel & ha).sum()
            tot["dis_" + nome + "_m"] += (dis & sel & hb).sum()
        for i, bonus in enumerate(BONUS):
            y = a.copy()
            y[righe, cb] += bonus
            f = y.argmax(1)
            verd[i] += (f == 0).sum()
            segue[i] += (f[dis] == cb[dis]).sum()
    n = tot["n"]
    i = int(verd.argmax())
    acc = {"persona": tot["h"] / n, "macchina": tot["mok"] / n, "coppia_ideale": tot["ideale"] / n, "pesi_uguali": tot["uguali"] / n,
           "pesa_il_verdetto": verd[i] / n, "macchina_tranne_molto_sicuro": tot["molto"] / n, "propria_tranne_poco_sicuro": tot["poco"] / n}
    out = {"scenario": sc, "peso_ideale": None if math.isinf(w_ideale) else w_ideale,
           "accuratezza": acc, "errori_su_cento": {k: 100 * (1 - v) for k, v in acc.items()},
           "sensibilita": {k: float(st.d_da_acc(v, m)) for k, v in acc.items()},
           "pesa_il_verdetto_bonus": float(BONUS[i]), "pesa_il_verdetto_segue_nei_disaccordi": segue[i] / tot["dis"],
           "allarme": {"disaccordi": tot["dis"] / n,
                       "errore_macchina_in_accordo": tot["m_err_acc"] / tot["acc"],
                       "errore_macchina_in_disaccordo": tot["m_err_dis"] / tot["dis"],
                       "quota_errori_macchina_in_disaccordo": tot["m_err_dis"] / max(tot["m_err_dis"] + tot["m_err_acc"], 1),
                       "nel_disaccordo_ragione_macchina": tot["dis_m"] / tot["dis"], "nel_disaccordo_ragione_persona": tot["dis_h"] / tot["dis"]},
           "sicurezza_nei_disaccordi": {nome: {"quota_dei_disaccordi": tot["dis_" + nome] / tot["dis"],
                                               "ragione_persona": tot["dis_" + nome + "_h"] / max(tot["dis_" + nome], 1),
                                               "ragione_macchina": tot["dis_" + nome + "_m"] / max(tot["dis_" + nome], 1)}
                                        for nome in ("molto", "poco")}}
    forte = max(out["sensibilita"]["persona"], out["sensibilita"]["macchina"])
    out["sopra_il_piu_forte"] = {k: v / forte - 1 for k, v in out["sensibilita"].items()}
    return out


def main():
    res = {}
    for nome, sc in SCENARI.items():
        res[nome] = corri(sc, SEME + 10)
        if nome in ("gpt4o", "pari"):
            res[nome + "-due-opzioni"] = corri(scenario_binario(sc), SEME + 11)
        print(nome, "done", flush=True)
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(st.HERE, "results", "rules.json"), "w"), indent=1, default=float)
    print("\nTHE ALARM (model)")
    print(f"{'scenario':<16}{'disagr.':>8}{'err.M agr.':>11}{'err.M dis.':>11}{'err. share in dis.':>20}{'right M':>8}{'right H':>8}")
    for nome, r in res.items():
        a = r["allarme"]
        print(f"{nome:<16}{100 * a['disaccordi']:>7.1f}%{100 * a['errore_macchina_in_accordo']:>10.1f}%{100 * a['errore_macchina_in_disaccordo']:>10.1f}%"
              f"{100 * a['quota_errori_macchina_in_disaccordo']:>19.0f}%{100 * a['nel_disaccordo_ragione_macchina']:>7.0f}%{100 * a['nel_disaccordo_ragione_persona']:>7.0f}%")
    print("\nTHE RULES: correct answers per hundred (model)")
    chiavi = ("persona", "macchina", "coppia_ideale", "pesi_uguali", "pesa_il_verdetto", "macchina_tranne_molto_sicuro", "propria_tranne_poco_sicuro")
    print(f"{'scenario':<16}" + "".join(f"{LABELS[c][:15]:>16}" for c in chiavi) + f"{'follows (verd.)':>15}")
    for nome, r in res.items():
        print(f"{nome:<16}" + "".join(f"{100 * r['accuratezza'][c]:>16.1f}" for c in chiavi) + f"{100 * r['pesa_il_verdetto_segue_nei_disaccordi']:>14.0f}%")
    print("\nTHE RULES: sensitivity above the stronger of the two, in per cent (model)")
    print(f"{'scenario':<16}" + "".join(f"{LABELS[c][:15]:>16}" for c in chiavi))
    for nome, r in res.items():
        print(f"{nome:<16}" + "".join(f"{100 * r['sopra_il_piu_forte'][c]:>16.1f}" for c in chiavi))
    print("\nCONFIDENCE IN DISAGREEMENTS (model): who is right when the person is very or not very confident")
    for nome, r in res.items():
        s = r["sicurezza_nei_disaccordi"]
        print(f"{nome:<16} very confident: person {100 * s['molto']['ragione_persona']:.0f}%, machine {100 * s['molto']['ragione_macchina']:.0f}% "
              f"({100 * s['molto']['quota_dei_disaccordi']:.0f}% of disagreements) | not very confident: person {100 * s['poco']['ragione_persona']:.0f}%, "
              f"machine {100 * s['poco']['ragione_macchina']:.0f}% ({100 * s['poco']['quota_dei_disaccordi']:.0f}%)")


if __name__ == "__main__":
    main()
