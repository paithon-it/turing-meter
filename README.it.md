# turing-meter

> [English](README.md) · **Italiano**. Il progetto è scritto in inglese: questa è
> la traduzione del README; specifica, piano, protocolli e codice sono in inglese.

**Quanto una macchina potenzia una mente umana, e quanto di ciò che le due
potrebbero fare insieme si vede davvero?**

`turing-meter` è lo strumento di misura di
[*Intelligenza apparente*](https://paithon.it/intelligenza-apparente/), articolo di paithon.it
(doi:[10.5281/zenodo.24107649](https://doi.org/10.5281/zenodo.24107649)). L'articolo descrive
l'intelligenza che si vede in un lavoro come la composizione ortogonale di una
componente naturale e di una algoritmica, come la potenza attiva e quella reattiva
in un circuito a corrente alternata: `|S| = sqrt(P^2 + Q^2)`. Questo repository
legge quella legge come l'ideale di una persona che lavora con una macchina, e
misura quanto le coppie reali ci si avvicinano.

> Stato: **in progettazione** (novembre 2026). Ci sono la specifica, lo schema
> dei compiti, i calcoli di base, due analisi su dati pubblici e un primo
> esperimento ([`experiments/angle-between-errors/`](experiments/angle-between-errors/)); mancano il runner
> per la macchina e lo strumento per le sessioni con le persone. Vedi
> [`docs/PLAN.md`](docs/PLAN.md).

## I numeri

Le stesse decisioni si prendono tre volte: dalla persona da sola (`dH`), dalla
macchina da sola (`dM`), dalla persona con la macchina (`dS`). Ognuna è una
sensibilità `d'` della teoria della rilevazione del segnale: zero al caso, nessun
tetto.

- **L'intelligenza apparente** `|S| = sqrt(dH^2 + dM^2)` è il meglio che due menti
  possono fare insieme quando sbagliano su casi diversi e mettono in comune tutta
  la loro evidenza (Green e Swets 1966; Sorkin e Dai 1994). Le coppie di persone ci arrivano
  vicino senza raggiungerlo (Bahrami et al. 2010). Ortogonali, in
  questa lettura, vuol dire errori indipendenti.
- **Il fattore di intelligenza** `eta = dS / |S|`: quanta parte dell'intelligenza
  apparente si vede nel lavoro, come il fattore di potenza è la potenza attiva
  sull'apparente. 0 al caso, 1 all'ideale.
- **La quota del guadagno** `g = (dS - dH) / (|S| - dH)`: quanto del guadagno
  possibile la persona realizza, 0 se ignora la macchina. `eta` scende per
  aritmetica quando la macchina è più forte; il comportamento lo descrive `g`.
- **Il turing** (Tu) `= dS / dH`: le stesse persone sugli stessi casi, da sole e
  con la macchina. Una mente che lavora da sola vale 1 Tu: ogni mente è il
  campione di se stessa, come nei valori in per unità dell'elettrotecnica. Nella
  definizione non compare nessuna macchina; i valori appartengono a persone e
  casi. `Tu^2` è il peso dell'evidenza che servirebbe a un osservatore ideale per
  fare come la coppia, diviso per quello che gli servirebbe per fare come la
  persona.
- **I turing apparenti** `Tu_app = |S| / dH`: l'ideale nell'unità della persona,
  ciò che promettono le due letture separate, come i voltampere accanto ai watt
  del turing (i voltampere si leggono senza l'angolo, `|S|` lo contiene già). Per
  costruzione `Tu = eta * Tu_app`. Dipendono solo da `k = dM / dH` e dall'angolo
  `phi` qui sotto: `sqrt(1 + k^2 - 2k cos(phi)) / sin(phi)`, cioè `sqrt(1 + k^2)`
  a 90 gradi.
- **La direzione** `theta`, con `cos(theta) = dH / |S| = 1 / Tu_app`, la quota
  della persona nell'intelligenza apparente; a 90 gradi `theta = atan(dM / dH)`.
- **L'angolo fra le menti** `phi = arccos(rho)`, dove `rho` è la correlazione dei
  loro errori sugli stessi casi. 90 gradi è la legge di `|S|`; si misura ogni
  volta che ci sono i dati caso per caso, e cambia l'ideale.
- **L'angolo utile** `arccos(debole / forte)`: accanto a una macchina `k` volte
  più forte, una persona aggiunge qualcosa all'ideale solo se gli errori si
  incontrano a più di `arccos(1/k)`, 78,5 gradi per `k = 5`. Sotto quell'angolo
  l'ideale è il più forte da solo, perché lo strumento non ammette pesi negativi:
  è una scelta, non una legge. Se lavorare con una macchina chiuda l'angolo di una
  persona è la domanda del prossimo studio ([`docs/STUDY.md`](docs/STUDY.md)); che
  le persone prendano gli errori dell'AI con cui hanno lavorato è già noto
  (Vicente e Matute 2023; Glickman e Sharot 2024).

## Che cosa dicono i dati pubblici

Su 87 confronti di decisioni binarie da 27 articoli di una meta-analisi di
esperimenti che hanno misurato persone da sole, AI da sola e persone con l'AI
(Vaccaro, Almaatouq e Malone, 2024), con l'angolo fissato a 90 gradi perché quei
dati non hanno le risposte caso per caso:

- le persone realizzano in mediana **0,30** del guadagno che l'ideale darebbe
  loro (intervallo al 95% sugli articoli 0,19-0,43; 0,38 sui confronti che
  misurano le stesse persone), e le coppie mostrano 0,69 della loro intelligenza
  apparente;
- la quota del guadagno **non** dipende da chi è più forte (0,24 quando lo è la
  persona, 0,32 quando lo è la macchina). La quota dell'ideale scende (0,81
  contro 0,59), ma per aritmetica: una macchina più forte alza il tetto più in
  fretta di quanto le persone la seguano;
- 0,46 quando l'AI mostrava la sua confidenza, 0,20 quando mostrava solo il
  verdetto: una differenza fra studi, e fragile (0,33 contro 0,26 contando ogni
  articolo una volta);
- 31 coppie battono la migliore delle due menti da sola; fra 0 e 13 oltre il
  rumore, secondo quanti casi stanno dietro l'accuratezza dell'AI, che gli studi
  non dicono. Solo 3 superano la regola che seguono le coppie di persone.

Dettagli, limiti e la seconda analisi (perché agenti diversi non si mettono mai
su una sola scala latente) in [`analysis/`](analysis/).

## Il primo esperimento

[`experiments/angle-between-errors/`](experiments/angle-between-errors/) è l'esperimento dell'articolo: il codice, il
protocollo scritto prima di qualunque risultato (con le deviazioni, datate) e i
risultati derivati. Misura l'angolo fra gli errori di persone e macchine sugli
stessi casi, su quattro fonti pubbliche con le risposte caso per caso (ChatBench,
ChaosNLI su αNLI, HAIID, model-vs-human), contro 24 macchine; quelle che le fonti
non forniscono sono girate su una GPU da 24 GB. Che cosa regge meglio:

- **Lo strumento è centrato dentro il suo modello.** In simulazione ritrova il
  turing vero entro l'1% in 6 situazioni su 7, con intervalli prudenti:
  l'intervallo al 95% contiene il vero nel 98-100% dei casi.
- **Con GPT-4o le persone salgono, e la coppia frena la macchina.** Un gruppo di
  persone vale 1,65 turing, e la coppia resta sotto la macchina a qualunque
  morsetto (resa sulla macchina 0,76). Con Llama 3.1 8B, più debole, 1,40 turing,
  e la coppia supera la macchina interrogata da sola (1,19), non quella della
  conversazione (0,92). Sono analisi aggiunte dopo il protocollo.
- **L'angolo non è retto dappertutto.** Su αNLI e sul testo di HAIID sta sotto i
  90 gradi; su ChatBench non si distingue dal retto in 13 confronti su 20
  (mediane fra 59 e 81 gradi). Su αNLI si chiude con la crescita delle macchine,
  da 76 a 56 gradi.
- **La formula prevede il tetto di una coppia, non il suo guadagno**, in una
  prova debole.

Ogni risultato ha accanto quanto è solido, deciso prima dei numeri o aggiunto
dopo, nell'articolo e nel protocollo.

## Principi

- **Una mente contro se stessa.** Nessuna classifica di modelli, nessun numero
  unico per modello.
- **Il modello può sbagliare.** Una coppia sopra l'ideale, oltre il rumore e con
  l'angolo misurato, smentisce la composizione per quel tipo di lavoro.
- **Il circuito fa parte del risultato.** Chi prende la decisione finale, che cosa
  mostra la macchina, in che ordine: il protocollo è scritto in ogni rapporto.
- **Un turing non si dà mai da solo.** Accanto vanno il compito, le persone, la
  macchina, il circuito, `dH`, `k`, l'angolo e un intervallo. Va letto accanto a
  `k`: chi ricopia una macchina `k` volte più sensibile vale `k` turing, e né
  `eta` né `g` distinguono chi copia; ciò che la persona aggiunge si vede solo
  dove `Tu` supera `k`. Dettagli nella sezione 5.1 di
  [`docs/SPEC.md`](docs/SPEC.md).
- **Niente prezzo.** Il costo del lavoro è un'altra domanda, e resta fuori
  dall'unità.
- **Le persone sono pseudonime.** I tentativi cronometrati hanno un
  identificativo, un ruolo, una fascia di esperienza e un anno; mai un nome.

## Su che cosa si basa

- [Inspect](https://inspect.aisi.org.uk/) (UK AI Security Institute, licenza MIT)
  per far girare la macchina da sola in sandbox.
- La teoria della rilevazione del segnale per la scala e l'ideale; Geirhos,
  Meding e Wichmann (2020) per la coerenza degli errori fra persone e macchine.
- La misura congiunta di Luce e Tukey, e la critica di Michell alla psicometria,
  per le diagnosi.

## Come citare

endjneer (2026). *Intelligenza apparente*. Versione 1. Zenodo. https://doi.org/10.5281/zenodo.24107649

Codice e dati: https://github.com/paithon-it/turing-meter

## Licenze

- Codice con licenza MIT, © 2026 paithon.it.
- Il protocollo e i risultati derivati dell'esperimento (`experiments/angle-between-errors/PROTOCOL*.md`,
  `experiments/angle-between-errors/results/`) con licenza [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it);
  ciò che deriva da ChaosNLI resta sotto la sua,
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.it).
- I dati delle fonti non sono ridistribuiti: gli script li leggono dalle fonti,
  sotto le licenze dei loro autori.
- Il testo dell'articolo: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it).
