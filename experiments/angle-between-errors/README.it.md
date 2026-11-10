> [English](README.md) · **Italiano**

# L'angolo fra gli errori

Codice, protocollo e risultati dell'esperimento dietro l'articolo di paithon.it
[«Intelligenza apparente»](https://paithon.it/intelligenza-apparente/).
Questa cartella fa parte del repository turing-meter, lo strumento.

Si misura l'angolo fra gli errori di una persona e di una macchina sugli stessi
casi, il turing (quante volte la stessa persona distingue meglio il giusto dallo
sbagliato con la macchina accanto), e si controlla se la legge della coppia
ideale prevede ciò che una coppia può davvero fare. Ipotesi e criteri stanno in
`PROTOCOL.it.md` (la traduzione inglese è `PROTOCOL.md`), scritto prima di
guardare i dati; ciò che è cambiato dopo sta nella sua sezione «Deviazioni».

## Come citare

endjneer (2026). *Intelligenza apparente*. Versione 1. Zenodo.
https://doi.org/10.5281/zenodo.24107649

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
