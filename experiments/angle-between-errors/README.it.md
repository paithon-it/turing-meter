> [English](README.md) · **Italiano**

# L'angolo fra gli errori

Codice, protocollo e risultati dell'esperimento dietro l'articolo di paithon.it
[«Intelligenza apparente»](https://paithon.it/intelligenza-apparente/).
Questa cartella fa parte del repository turing-meter, lo strumento.

Si misura l'angolo fra gli errori di una persona e di una macchina sugli stessi
casi, il turing (quante volte la stessa persona riconosce meglio la risposta
giusta con la macchina accanto), e si controlla se la legge della coppia
ideale prevede ciò che una coppia può davvero fare. Ipotesi e criteri stanno in
`PROTOCOL.it.md` (la traduzione inglese è `PROTOCOL.md`), scritto prima di
guardare i dati; ciò che è cambiato dopo sta nella sua sezione «Deviazioni».
Le analisi da A13 ad A19 e gli esiti di A11 stanno solo in `PROTOCOL.md`, che per quelli fa fede.

## Come citare

endjneer (2026). *Intelligenza apparente* (*Apparent Intelligence*). Zenodo.
https://doi.org/10.5281/zenodo.24107648 (tutte le versioni; l'ultima è la versione in forma
di paper, in italiano e in inglese)

Codice e dati: https://github.com/paithon-it/turing-meter, cartella
`experiments/angle-between-errors`.

## Che cosa c'è

- `angle/`: stimatori (`sdt.py`, `measure.py`), fonti di dati (`sources.py`,
  `download.py`, `items.py`), test della legge (`combiner.py`), analisi
  (`analysis*.py`, `run_chatbench.py`, `vaccaro.py`), riassunto e figure.
- `gpu/`: gli script che fanno girare modelli e reti su una GPU.
- `tests/`: gli stimatori provati su dati simulati con il disegno di ciascuna
  fonte. Criterio del protocollo: errore sistematico di ρ sotto 0,03.
- `results/`: i risultati su cui è scritto l'articolo. Le risposte delle
  macchine stanno a parte (vedi sotto).

## Rifarlo

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas scikit-learn pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python -m angle.download data chatbench chaos haiid geirhos   # i dati pubblici, in data/
.venv/bin/python -m angle.items data                                    # i casi, in data/items/
```

I dati di ChaosNLI (CC BY-NC 4.0) e della meta-analisi di Vaccaro non si
ridistribuiscono. ChaosNLI lo scarica `angle/download.py`; la tabella di
Vaccaro, Almaatouq e Malone (2024) va presa a mano dal repository OSF degli
autori (https://osf.io/wrq7c/?view_only=b9e1e86079c048b4bfb03bee6966e560, il
link pubblicato nel loro articolo) e salvata come `data/vaccaro/Data_Extraction.csv`. Gli script non
eseguono codice degli autori dei dati.

Sulla GPU (NVIDIA da 24 GB, vLLM e PyTorch), dalla cartella dell'esperimento:

```bash
bash gpu/prepare.sh                     # modelli, immagini e dati, in ~/rnd
bash gpu/queue.sh                       # reti per la visione e persone di silicio
bash gpu/run_llm.sh                     # modelli linguistici
```

Le risposte finiscono in `gpu-results/` (non sono in questo repository: si rifanno con gli script di `gpu/`). Poi, in locale:

```bash
.venv/bin/python -m angle.analysis         CARTELLA_RISULTATI_GPU
.venv/bin/python -m angle.analysis_vision  CARTELLA_RISULTATI_GPU
.venv/bin/python -m angle.analysis_silicon CARTELLA_ESITI_SILICIO
.venv/bin/python -m angle.run_chatbench
.venv/bin/python -m angle.vaccaro data/vaccaro/Data_Extraction.csv
.venv/bin/python -m angle.summary
.venv/bin/python -m angle.figure
```

Le due prove dello strumento (protocollo, «Lo strumento alla prova»):

```bash
.venv/bin/python -m angle.calibration                          # simulazione, una ventina di minuti
.venv/bin/python -m angle.full_evidence CARTELLA_RISULTATI_GPU # le dieci reti per la visione
```

Le quattro prove chieste da una revisione il 10 novembre 2026, esplorative
(protocollo, «Deviazioni del 10 novembre 2026»), scrivono `results/review.json`
senza toccare gli altri risultati:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m angle.review CARTELLA_RISULTATI_GPU   # pochi minuti con due processi
```

`summary.json` ha una riga per ipotesi del protocollo.

## I modelli chiusi (aggiunta del 14-15 novembre 2026)

Quattro modelli Claude (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.2) hanno
risposto da soli alle domande di ChatBench e αNLI, una domanda per chiamata,
attraverso Claude Code in modalità headless (`claude -p`: nessuno strumento,
nessun server MCP, nessuna impostazione ereditata, una cartella di lavoro vuota,
la lettera vincolata alle opzioni da uno schema di risposta). Sui modelli più
grandi il ragionamento non si spegne, quindi tutti e quattro hanno girato con
impegno `low`. Non ci sono probabilità, solo verdetti, e le persone non hanno mai
lavorato con questi modelli: la corsa dà k, l'angolo e il margine della coppia
ideale sopra la macchina, non un turing. Esplorativa (protocollo, «Deviazioni
del 14 novembre 2026»). Le risposte stanno in `closed-results/`, perché non si
possono rifare uguali.

```bash
python -m closed.machines_claude probe claude-haiku-5-5                              # che cosa riceve il modello
python -m closed.machines_claude run claude-opus-5-5 opus-5.5 chatbench,alphanli low  # riprende da dove si è fermato
.venv/bin/python -m angle.frontier closed-results CARTELLA_RISULTATI_GPU              # scrive results/frontier.json
```

## Dentro le conversazioni (aggiunta del 15 novembre 2026)

Che cosa succede nelle conversazioni di ChatBench, esplorativo e chiesto da due
revisioni (protocollo, «Deviazioni del 15 novembre 2026»): dove vanno i punti fra
la macchina da sola e la coppia, estrazione e filtro, quanto spesso le persone
seguono la chat quando ha ragione e quando ha torto (contro un osservatore
lineare senza gating), chi la segue sempre e chi mai contro persone tutte uguali,
le persone divise per bravura (mediana su 300 divisioni a metà, in tutto e nella
sola matematica), e il limite della regola del tre per un modello senza errori.

```bash
.venv/bin/python -m angle.conversations     # scrive results/conversations.json, circa mezzo minuto
```

## Dopo la sesta revisione (aggiunta del 15 novembre 2026)

Due analisi chieste da una sesta revisione, esplorative (protocollo, «Dopo la
sesta revisione»). Il tetto di una coppia quando uno dei due dà solo un
verdetto: l'osservatore ottimo nel modello dell'articolo, per le righe della
tabella del margine e per le due coppie di ChatBench. E, su αNLI, la vicinanza
fra le persone e una macchina al netto della difficoltà per una terza
popolazione (le macchine aperte), e fra due macchine al netto della difficoltà
per le persone, con le storie contestate definite su metà dei voti.

```bash
.venv/bin/python -m angle.verdict                                           # scrive results/verdict-ceiling.json, circa venti minuti
.venv/bin/python -m angle.difficulty closed-results CARTELLA_RISULTATI_GPU  # scrive results/difficulty.json, circa dieci minuti
```

## Licenze

- Codice (`angle/`, `gpu/`, `tests/`): licenza MIT, vedi il `LICENSE` del
  repository in `../../LICENSE`.
- Protocollo e risultati derivati (`PROTOCOL.md`, `PROTOCOL.it.md`, `results/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it). Ciò che
  deriva da ChaosNLI resta sotto la sua licenza,
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.it).
- I dati delle fonti non stanno qui: ChatBench (CC BY 4.0), ChaosNLI
  (CC BY-NC 4.0), HAIID (MIT), model-vs-human e la meta-analisi di Vaccaro
  restano sotto le licenze dei loro autori.
- Il testo dell'articolo: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it).
