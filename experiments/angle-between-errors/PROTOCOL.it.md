# Protocollo: l'angolo fra gli errori, misurato

> Testo originale, in italiano, scritto prima dei risultati. Rispetto all'originale sono cambiati soltanto i nomi di file, moduli, funzioni e chiavi, ora in inglese come il codice; la traduzione è in [`PROTOCOL.md`](PROTOCOL.md).

Scritto il 9 novembre 2026 **prima** di calcolare qualunque risultato. Serve a
non scegliere le analisi dopo aver visto i numeri. Ciò che cambia dopo si
segnala in fondo, nella sezione «Deviazioni», con il motivo.

## Il problema

L'articolo 534 definisce l'angolo φ fra gli errori di una persona e quelli di
una macchina, poi dichiara di non poterlo misurare e lo fissa a 90 gradi. Tutte
le sue cifre dipendono da quella ipotesi, compresa «le persone realizzano un
terzo del guadagno possibile». Qui l'angolo si misura, su dati pubblici con le
risposte caso per caso, e si controlla se la legge predice davvero quello che
una coppia può fare.

Non si misurano persone nuove: le risposte umane sono pubbliche. Le macchine
che le fonti non forniscono girano su una GPU NVIDIA L4 da 24 GB.

## Modello e definizioni

Teoria della rilevazione del segnale, rumore gaussiano a varianza uguale.

- **d′**. Sì/no: `2 z(a)` quando le classi sono bilanciate e il criterio neutro,
  altrimenti `z(hit) − z(fa)`. Scelta forzata fra `m` opzioni: si inverte
  `PC = ∫ φ(x − d) Φ(x)^(m−1) dx`. I rapporti (turing, η, g) non dipendono dal
  fattore di scala.
- **ρ, φ**. ρ è la correlazione fra l'evidenza della persona e quella della
  macchina sugli stessi casi, condizionata alla risposta giusta; φ = arccos ρ.
  La difficoltà condivisa di un caso **conta** in ρ, perché è proprio ciò che
  rende inutile il secondo parere. Con tante persone e pochi casi a testa si
  stima con la soglia personale di ciascuno (`rho_personal_thresholds`), ottenuta
  dalla sua accuratezza con un restringimento empirico-bayesiano.
- **Ideale** `|S|(φ) = sqrt((dH² + dM² − 2ρ dH dM) / (1 − ρ²))`, fermato al più
  forte quando `ρ` supera `min/max` (niente pesi negativi).
- **Angolo utile** `arccos(min(dH, dM) / max(dH, dM))`.
- **g(φ)** `= (dS − dH) / (|S|(φ) − dH)`, la quota del guadagno possibile.

Con più di due opzioni la correlazione delle risposte giuste non è quella
dell'evidenza: la si converte con una tabella calcolata per simulazione, per le
sensibilità di quel confronto. La conversione si verifica su dati simulati.

## Dati

| Sigla | Fonte | Compito | Persone | Casi | dS misurato |
|---|---|---|---|---|---|
| CB | ChatBench (Chang, Anderson, Hofman 2025), CC BY 4.0 | MMLU, 4 opzioni, matematica/fisica/etica | sì, a coppie persona-domanda | 396 | sì, condizione *answer-first* |
| CN | ChaosNLI, αNLI (Nie, Zhou, Bansal 2020) | 2 opzioni, abduzione, verità dalla storia originale | 100 voti per caso | 1.532 | no |
| HA | HAIID (Vodrahalli et al. 2022) | 5 compiti binari, risposta graduata | stessa persona prima e dopo | 152 | sì |
| GE | model-vs-human (Geirhos et al.) | 16 categorie, 17 condizioni | prova per prova | ~21.000 prove | no |

Macchine: ChatBench fornisce GPT-4o e Llama 3.1 8B da sole. Sulla L4 girano modelli
aperti di generazioni e famiglie diverse sulle stesse domande (probabilità di
ogni opzione, senza ragionamento) e per la visione una scala di reti dal 2012
in poi. Per HAIID, modelli di visione e di linguaggio sui casi originali.

## Ipotesi e criteri

Fissate prima. «Passa» vuol dire che l'intervallo al 95 per cento rispetta il
criterio.

- **H1, l'angolo non è retto.** φ persona-macchina sotto i 90 gradi su ogni
  famiglia di dati. Si riportano φ e intervallo per ciascun confronto.
- **H2, le macchine si somigliano.** ρ macchina-macchina maggiore di ρ
  persona-macchina. Replica di risultati noti.
- **H3, la legge predice.** *Il test principale.* Su ogni confronto si addestra
  un combinatore (regressione logistica, validazione incrociata a cinque parti
  per caso) che usa solo ciò che la coppia ha davvero: risposta ed eventuale
  sicurezza della persona, probabilità della macchina. L'errore assoluto medio
  fra il d′ del combinatore e la previsione con il φ misurato deve essere
  minore di quello della previsione a 90 gradi. Con una persona che dà solo il
  verdetto la previsione è l'ideale *a verdetto* con φ, calcolato per
  simulazione: la legge a evidenza piena sarebbe un bersaglio irraggiungibile.
- **H4, il terzo si sposta.** Dove dS è misurato (CB, HA), g(φ misurato) è
  maggiore di g(90°). La previsione è un aumento, l'entità è aperta. Sulla
  meta-analisi di Vaccaro si riporta g per ρ da 0 a 0,6 come analisi di
  sensibilità, perché lì ρ non si può misurare.
- **H5, l'angolo utile.** Per ogni coppia si dice se φ supera l'angolo utile, e
  se il combinatore fa meglio della macchina da sola oltre l'errore. Le due
  risposte devono concordare; le discordanze si riportano.

## Stima e incertezza

Bootstrap a due livelli (casi e persone) con 2.000 ripetizioni, intervalli
percentili. Si riportano mediane e intervalli, non significatività.
Dove la macchina dà 50 risposte per caso (GPT-4o e Llama da ChatBench) la sua
«risposta giusta» in un caso è la frazione di campioni giusti.

Prima di toccare i dati veri, gli stimatori passano una prova su dati simulati
con il disegno di ciascuna fonte: errore sistematico di ρ inferiore a 0,03 in
valore assoluto; altrimenti si corregge lo stimatore. Quel controllo è in
`tests/` e gira a ogni modifica.

## Persone di silicio: la prova del protocollo di assimilazione

Un nostro protocollo precedente, non pubblicato, propone la misura
`A = ρHM − sqrt(q ρHH)` per separare gli errori presi dalla macchina dal semplice
effetto della pratica, ma è stato provato solo su un generatore gaussiano fatto
apposta. Qui lo si prova con reti vere su immagini vere, dove la difficoltà
condivisa non è un singolo fattore.

Sessanta reti piccole («persone») imparano CIFAR-10 a tappe, in tre bracci:
**controllo** (la risposta giusta dopo ogni prova), **assistito neutro** (in più
il parere di una macchina congelata, senza difetti iniettati) e **assistito con
difetto** (la stessa macchina con un errore sistematico iniettato su due classi).
Sulle stesse immagini mai viste, alle tappe T0, T1, T3, si misura ciascuna rete
da sola.

Si dichiara in anticipo che cosa vorrebbe dire che la misura funziona:
- **Falso allarme**: nel confronto controllo contro assistito neutro, con bracci
  di bravura uguale, la differenza nelle differenze di A deve avere un intervallo che comprende lo zero.
- **Sensibilità**: nel braccio con difetto, A deve crescere più del controllo.
- **La controprova**: ρHM ingenua deve crescere con la pratica anche nel
  controllo, cioè il trabocchetto per cui A esiste deve presentarsi davvero.

Se la misura fallisce una di queste prove, va scritto così: una rete non è una
persona, ma una misura che fallisce su un caso noto non è pronta per le persone.

## Che cosa non si può concludere

Nessuna delle fonti segue le persone nel tempo: la domanda sull'assimilazione
resta aperta. I dati umani sono crowdworker su compiti brevi, non specialisti. Le macchine
che le fonti forniscono sono quelle del 2024-2025. Un angolo misurato vale per
quel compito, quelle persone e quella macchina.

## Deviazioni

Registrate man mano, con la data e il motivo. Tutto ciò che sta qui sotto è
stato deciso **dopo** il primo sguardo ai dati, e va letto come esplorazione.

- **9/11/2026, dopo i primi numeri di ChatBench (φ di GPT-4o e Llama con la
  macchina da sola).** Con Llama la coppia supera l'ideale (η > 1): la macchina
  «da sola» di ChatBench è un modello interrogato senza la persona, mentre in
  chat la persona chiede, corregge, insiste. La macchina fissa è il modello
  sbagliato: si misurano ai morsetti sbagliati. Si aggiungono:
  - **M-in-chat**: per ChatBench anche la risposta che la macchina ha dato
    *in quella conversazione* (`answer_letter` di `conversations.json`, dove
    c'è), macchina per prova, con le stesse formule. Si riportano entrambe.
  - **H6, chi decide prima.** ChatBench assegna a caso se la persona risponde
    da sola prima di interrogare la macchina (*answer-first*) o se va subito da
    lei (*direct-to-AI*). Si confrontano accuratezza e d′ finale nelle due
    condizioni, per macchina: è il solo confronto randomizzato qui dentro
    sull'ordine di chi prende l'iniziativa.
  - **H7, la scala.** Sulle stesse persone e le stesse domande di ChatBench, le
    macchine dal modello piccolo al grande: come cambiano φ, l'angolo utile, il
    turing possibile e cos θ (la parte della persona nell'ideale) quando la
    macchina cresce. È la prova dell'ipotesi che la crescita della tecnica
    non avvicini la macchina all'umano, e dice da sé che cosa la falsificherebbe.
- **9/11/2026, leggendo il metodo di HAIID.** Il «consiglio» mostrato ai
  partecipanti, presentato ora come AI ora come persona, era la **risposta media
  di una cinquantina di persone** (Vodrahalli et al., sez. 3.1), con una piccola
  perturbazione casuale: non è una macchina. Non si usa come macchina: resta,
  con questa etichetta, come controllo («la folla»), per sapere che angolo e che
  quota dà un parere umano. Le macchine di HAIID sono solo quelle fatte girare
  sulla L4 sui casi originali.
- **9/11/2026, a esperimenti fatti, sulle persone di silicio.** Il primo calcolo
  del criterio di sensibilità confrontava due cose diverse: il braccio con
  difetto era misurato contro la macchina difettosa, il controllo contro quella
  sana. Il confronto corretto usa lo stesso comparatore per i due: il controllo si
  misura anche contro la macchina difettosa. Si riportano tutti e due i numeri, e
  in più una misura diretta che non dipende da A: sui soli casi in cui il difetto
  cambia la risposta, quante volte la rete dà proprio la risposta sbagliata
  iniettata.
- **9/11/2026, dopo una revisione indipendente del codice**, tutte correzioni di
  errori nostri trovati dopo aver visto i numeri:
  - *Il difetto del silicio.* Era estratto a caso a ogni passaggio, quindi non
    dipendeva dall'immagine: una rete poteva ereditare solo una tendenza di classe,
    che la misura A toglie per costruzione. La prima corsa non poteva dire niente
    sulla sensibilità di A. Si è rifatta con un difetto che dipende solo
    dall'immagine (una direzione fissa delle caratteristiche della macchina), e si
    riporta A anche senza condizionare sulla classe.
  - *Il bootstrap persona-persona.* Duplicare le righe accoppiava una prova con la
    sua copia e gonfiava l'intervallo: ora il peso sta sul caso. La stima puntuale
    non cambia.
  - *Il d′ con il criterio spostato.* Il protocollo chiedeva z(hit) − z(fa) quando
    il criterio non è neutro; il codice usava 2 z(a) per tutti, e alcuni modelli
    su HAIID e αNLI pendono molto da una parte. Ora persona, macchina e combinatore
    usano il criterio libero, con la correzione di ½. Il combinatore vede anche
    quale opzione ha scelto la macchina.
  - *La conversione con m > 2.* Era calcolata su persone tutte uguali e superava
    il criterio di errore del protocollo su ChatBench. Ora simula le persone con
    la loro dispersione di sensibilità e la soglia stimata da poche risposte.
  - *La griglia di ρ.* Faceva da tetto (0,97) e da pavimento (−0,6). Ora va da
    −0,9 a 0,995; i valori al limite si contano.
  - *H3.* Con ρ > 0 la previsione a 90 gradi sta sopra quella con l'angolo, quindi
    quando il combinatore sta sotto tutte e due l'angolo vince senza merito. Si
    riportano anche le righe informative, quelle in cui il combinatore cade fra le
    due previsioni, e il guadagno sul migliore dei due.
  - *L'intervallo di H5* ricampiona ora anche le persone.
- **9/11/2026, il d′ del combinatore.** Per coerenza con persona e macchina il d′
  della coppia è calcolato classe per classe, con criterio libero. La correzione
  di ½ va fatta sui casi distinti, come per la macchina: sulle righe persona ×
  caso esplodeva quando il combinatore imparava a dire sempre la stessa classe su
  una ventina di casi. Su HAIID i casi per compito sono 24-32, e i d′ della
  macchina toccano spesso il massimo: il test della legge lì è debole, e si
  riporta per famiglia.
- **9/11/2026, dopo la seconda corsa del silicio.** Con il difetto legato
  all'immagine, A dentro la classe, come prescritta, fallisce ancora la prova di
  sensibilità, mentre la copia diretta del difetto si vede. Per capire il perché
  si sono aggiunte due analisi, entrambe esplorative: A senza condizionare sulla
  classe e A sulle sole due classi su cui il difetto agisce. Il criterio dichiarato
  in anticipo resta fallito.
- **9/11/2026, dopo una seconda revisione indipendente:**
  - *Il confronto ai due morsetti di ChatBench.* La riga «da sola» e la riga «in
    chat» usavano prove diverse: la seconda tiene solo le conversazioni con una
    risposta registrata, e le altre avevano un'accuratezza molto più bassa. Si
    aggiunge la macchina da sola calcolata sulle stesse prove della chat, ed è
    quella che si confronta.
  - *Il bootstrap del silicio* ricampionava le reti con reinserimento e faceva
    coppia fra una rete e la sua copia, gonfiando ρHH: ora il peso sta sulla rete.
  - *Il test della legge.* Le righe con combinatore e previsioni identiche (al
    tetto dell'accuratezza) erano contate due volte fra «sotto» e «sopra»: ora
    sono a parte. Si riportano i totali anche senza HAIID, perché lì il d′ del
    combinatore è a criterio libero su 16 casi per classe e non è confrontabile
    con la previsione simulata. La scelta è fatta dopo aver visto i numeri.
- **9/11/2026, messa a punto delle macchine.** Qwen3.8 27B sta nei 24 GB solo
  con più memoria assegnata e un contesto più corto. Sulle foto di dermatologia di
  HAIID i modelli visivi piccoli (e Qwen3.8) lavorano sull'immagine ridotta a 1024
  pixel per lato: la risoluzione originale non entra nel loro contesto.
- **9/11/2026, bootstrap.** Il protocollo diceva 2.000 ripetizioni. Si usano per
  tutti i confronti di testo e di HAIID; per la visione 600 (689 confronti con m
  = 16) e per il silicio 300, per il costo di calcolo. Sono intervalli meno
  stabili, e non cambiano le stime puntuali.
- **9/11/2026, la visione nel test della legge.** Con i dati di Geirhos la
  persona dà solo un verdetto: il combinatore usa solo la probabilità della
  macchina, e la previsione è l'ideale a verdetto.
- **9/11/2026, terza revisione: il turing come unità.** Aggiunte descrittive,
  decise dopo aver visto i numeri; nessuna ipotesi nuova e nessun criterio.
  - *Turing apparenti.* `|S| / dH` (`sdt.apparent_turings`): l'ideale della
    coppia nell'unità della persona. Dipende solo da k = dM/dH e da φ; il turing
    misurato è η volte questo. Nella tabella di ChatBench prende il posto della
    colonna di g, che resta nel testo.
  - *Gli intervalli del turing.* Il bootstrap a due livelli li calcolava già
    (`turing_ci`) e l'articolo non li riportava: ora accompagnano ogni valore,
    anche per materia (matematica, fisica, etica).
  - *Il turing su altre fonti.* Sui confronti della meta-analisi di Vaccaro
    (`vaccaro.turing_of_comparisons`: mediana, intervallo ricampionando gli
    studi, quanti sotto uno) e sul consiglio della folla di HAIID (già in
    `map.json`). Serve a mostrare che l'unità si applica a macchine di altri
    anni e a un aiuto che non è una macchina; non prova niente sull'angolo.
- **9/11/2026, dopo una terza revisione indipendente**, correzioni di errori
  nostri e aggiunte descrittive:
  - *La sensibilità della coppia con due classi.* Persona e macchina avevano il
    d′ a criterio libero, la coppia `2 z(a)`: ora la coppia usa la stessa
    convenzione. Tocca solo le cinque righe del consiglio della folla su HAIID
    (le uniche con due classi e una risposta di coppia): il turing del sarcasmo
    passa da 1,31 a 1,33, quello del censimento da 1,16 a 1,16 con un intervallo
    che ora esclude l'uno. Il resto è invariato.
  - *L'angolo fra due persone su ChatBench.* Era calcolato sulle risposte giuste
    senza la conversione per le quattro opzioni, e confrontato con quello fra
    persona e macchina, che è convertito. Ora è convertito anche lui (due
    persone con la sensibilità media del gruppo: un'approssimazione): da 69 a 66
    gradi. La conclusione non cambia, e si rafforza.
  - *Il turing letto dalla parte della macchina.* Si aggiunge `dS / dM`, con il
    suo intervallo dallo stesso bootstrap: dice se la persona, accanto alla
    macchina, aggiunge o toglie. E si riporta che cosa avrebbe ottenuto chi
    avesse ricopiato la macchina (`dS = dM`), perché né η né g lo distinguono da
    una coppia ideale vicino all'angolo utile.
  - *Un'altra scala.* Il turing ricalcolato sulla scala logistica del modello di
    scelta di Luce, per dire di quanto dipende dall'ipotesi gaussiana.
  - *Il tetto di una folla.* Dall'angolo fra due persone, il limite a cui arriva
    una folla di pari che mette in comune gli indizi: `1 / sqrt(ρ)`.

## Lo strumento alla prova (aggiunta del 9 novembre 2026, sera)

Scritta **prima** di calcolare queste due prove, ma dopo aver visto tutti i
risultati qui sopra: sono prove dell'unità di misura, non dell'angolo. Non
usano macchine nuove: la prima è una simulazione, la seconda lavora sulle
probabilità delle dieci reti per la visione già salvate.

### T, la taratura

Uno strumento va provato dove la verità si conosce. Si generano dati dal
modello (indizi gaussiani per opzione, quattro opzioni) con il disegno di
ChatBench: 160 persone con sensibilità diverse, 7 domande a testa su 380, una
macchina con sensibilità `dM` e correlazione `ρ` con le persone, e una coppia
che somma gli indizi della persona e quelli della macchina con un peso `w`. Il
valore vero di ogni grandezza è quello che lo stesso strumento restituisce su
un campione enorme dello stesso processo.

Scenari: (A) coppia ideale; (B) coppia a metà strada; (C) macchina debole
seguita troppo, turing sotto uno; (D) la persona copia la macchina, turing
uguale a k; (E) macchina pari alla persona; (F) persona debole e trenta persone
sole, come in fisica; (G) come A, con la macchina che risponde prova per prova.
Per ciascuno almeno 150 repliche, e su ogni replica lo stesso `compare`
dell'articolo con 300 ricampionamenti.

Si riportano, per turing, turing apparenti, η e φ: errore sistematico, errore
quadratico medio e copertura dell'intervallo al 95 per cento.

**Criterio.** Per il turing, negli scenari da A a E e in G: errore sistematico
entro il 5 per cento del valore vero e copertura fra il 90 e il 98 per cento.
Nello scenario F ci si aspetta di peggio, e si riporta di quanto. Se il
criterio fallisce l'articolo lo dice, e gli intervalli del turing si leggono di
conseguenza.

### E, la legge a evidenza piena

Il test H3 lavorava con persone che danno solo un verdetto. Le dieci reti
(2014-2023) hanno lasciato, per ciascuna delle 18.080 immagini, la probabilità
di ogni categoria: fra due reti l'evidenza è piena da tutte e due le parti, ed
è il caso in cui la formula dell'ideale dovrebbe valere. In ogni coppia di reti
la più debole fa da «mente» e la più forte da «macchina».

Per ogni coppia e ogni condizione (le stesse dell'analisi della visione, senza
le *cue-conflict*): `dA` e `dB` dall'accuratezza con sedici opzioni; `ρ` con lo
strumento dell'articolo (tetracorica delle risposte giuste, convertita); la
previsione `|S|` con l'angolo misurato e a 90 gradi; e la sensibilità raggiunta
da un combinatore a evidenza piena, che somma i logaritmi delle due
probabilità con due pesi appresi e si valuta su immagini non viste. Si
escludono le condizioni in cui una delle due è al livello del caso, e si
contano a parte quelle in cui la coppia tocca il tetto dell'accuratezza.

**Criteri.** (1) L'errore assoluto medio della previsione con l'angolo misurato
è minore di quello a 90 gradi, con l'intervallo al 95 per cento della
differenza, ricampionando le condizioni, tutto sotto lo zero. (2) Il rapporto
mediano fra ciò che il combinatore raggiunge e l'ideale con l'angolo sta fra
0,9 e 1,1. Se (2) fallisce verso l'alto, la formula sottostima ciò che una
coppia a evidenza piena può fare, e va scritto.

Si riporta anche, descrittiva, la tabella «una mente del 2014 accanto alle
macchine venute dopo»: VGG16 come mente, le altre nove come macchine, con k, φ,
turing apparenti, turing del combinatore e η; e, per famiglia di degradazione,
come cambiano turing e guadagno in accuratezza al crescere della difficoltà.

**Limiti dichiarati prima.** Una rete non è una persona. Coppie e condizioni
non sono indipendenti: le reti si somigliano, e ogni rete entra in nove coppie.
Le probabilità salvate sono arrotondate a cinque decimali. A evidenza piena il
modello gaussiano è un'approssimazione, e una rete troppo sicura di sé lo viola.

### Esiti e deviazioni delle due prove (9 novembre 2026, notte)

**T, la taratura.** Il criterio **fallisce** in cinque scenari su sei (A, B, C,
E, G), e dalla parte prudente: l'errore sistematico del turing sta entro l'1
per cento, ma l'intervallo al 95 per cento contiene il vero 98-100 volte su
cento, perché è da 1,3 a 1,7 volte più largo del necessario. Passa solo D, sul
bordo (98 per cento). In F l'errore sistematico è +4,1 per cento e l'errore
quadratico medio il 19 per cento del vero. Gli intervalli del turing
dell'articolo vanno letti come intervalli al 99 per cento circa. Il meccanismo
non è stato isolato.

Deviazioni: 300 repliche per A e B, 150 per gli altri (la macchina era carica;
il minimo dichiarato era 150); il seme del ricampionamento cambia a ogni
replica, altrimenti tutte avrebbero condiviso gli stessi pesi; il valore vero
viene dalle accuratezze su due milioni di prove con il ρ generativo (nel JSON
c'è anche lo strumento stesso su un milione di prove, come dice la lettera del
protocollo: per il turing coincide); la sensibilità minima di una persona è
tagliata a 0,02.

Difetti dello strumento trovati dalla taratura, **dichiarati e non ancora
corretti**, perché correggerli chiede di rifare tutte le corse:
- la correlazione si calcola su una griglia di soglie a passo 0,02
  (`measure._fast_rho`, e la griglia in `sdt.estimate_from_evidence`): con dati
  senza fine ρ non converge al vero (0,16 contro 0,20 nello scenario A). Sulle
  repliche l'errore ha media circa zero; sui dati veri di ChatBench sposta
  l'angolo di meno di un grado (GPT-4o 74,5 contro 75,2 gradi);
- `measure._sd_sensitivity` e `sdt.shrink` sottostimano la dispersione delle
  sensibilità di un fattore (1 − 1/n); l'effetto su ρ è trascurabile.

**E, la legge a evidenza piena.** I due criteri **passano**: errore assoluto
medio 0,10 con l'angolo misurato contro 0,75 a 90 gradi (differenza −0,65,
intervallo da −0,75 a −0,54, ricampionando 76 condizioni), e rapporto mediano
con l'ideale 1,00. Passano però con poco merito, e l'articolo lo scrive: fra
queste reti l'angolo mediano sta fra 22 e 43 gradi, 1.383 unità su 2.615 cadono
sotto l'angolo utile, e lì la previsione coincide con la rete più forte. Una
coppia che ricopia la più forte passa gli stessi criteri con un errore minore
(0,07). Sopra l'angolo utile la legge prevede un guadagno mediano dell'1,3 per
cento, il combinatore non ne ottiene, e la correlazione di rango fra guadagno
previsto e misurato è 0,09. Ciò che la prova sostiene è l'angolo utile, non la
legge del guadagno.

Deviazioni: 77 condizioni invece delle 73 di `vision.json` (ristretto alle 73
il risultato non cambia); 63 unità con la coppia al livello del caso fuori
campione, stato non previsto, tenute a parte; la conversione di ρ è quella di
`sdt.convert_rho_mafc` ricalcolata senza arrotondamenti. La linea di base «chi
ricopia la macchina» è stata aggiunta durante il lavoro.

Difetto trovato, non corretto: `sdt.convert_rho_mafc` ha la griglia fino a
0,95 e non rende monotona la tabella; non è usata dalle analisi dell'articolo
tranne che per l'angolo fra due persone su ChatBench, dove ρ è lontano dal
bordo.

## Deviazioni del 10 novembre 2026

- **10/11/2026, dopo una revisione.** Una revisione ha chiesto
  quattro prove di robustezza. Sono state decise dopo aver visto tutti i
  risultati, quindi sono **esplorative**: non cambiano ipotesi, criteri né
  numeri già scritti, e un esito favorevole non vale come conferma. Il motivo
  di ciascuna è un'ipotesi alternativa che i risultati riportati non
  escludevano. Codice in `angle/review.py` (nessuna funzione esistente è
  cambiata), numeri in `results/review.json`, prove del codice nuovo in
  `tests/test_review.py`. Il calcolo intero dura circa tre minuti.
  - *La scala su αNLI senza le chiavi contestate.* Motivo: crescendo, le
    macchine potrebbero concentrare gli errori residui sui casi in cui i
    votanti contestano la chiave, e la tetracorica, che suppone indizi
    gaussiani, leggerebbe quella concentrazione come correlazione. Stesso
    stimatore (`compare_votes`, 2.000 ricampionamenti), stesse dieci macchine,
    stesse due modalità: (a) senza i 157 casi su 1.532 in cui la maggioranza
    dei 100 voti va contro la chiave (gli 8 casi pari restano: 1.375 casi);
    (b) con l'opzione più votata come chiave (senza gli 8 pari: 1.524 casi).
    Sui casi originali la stessa corsa ridà `map.json` alla cifra. Esito: la
    discesa resta, più corta. Spearman fra dM e φ −0,89 e −0,90 in (a), −0,93
    e −0,94 in (b), contro −0,96 e −0,95 (risposta secca e ragionamento). Tutti
    gli angoli salgono, di 6-17 gradi, e la distanza fra la macchina più debole
    e la più forte scende in (a) da 21 a 15 gradi in risposta secca (da 76 a 55
    gradi prima, da 83 a 68 ora) e da 20 a 12 con il ragionamento. Una parte
    della discesa, dal 30 al 40 per cento, viene dai casi contestati; il resto
    no. In (b) la
    chiave è definita dagli stessi voti con cui si misura la persona, e
    l'accuratezza umana è gonfiata per costruzione (0,88 contro 0,84).
  - *L'angolo utile con pesi liberi.* Motivo: in E il combinatore ha pesi non
    negativi, quindi sotto l'angolo utile non può superare la rete più forte se
    non per rumore, e la conferma dell'angolo utile poteva venire dal vincolo.
    Stesse 2.615 unità valide, stessa divisione in cinque parti, stesso ε; pesi
    fra −1.000 e 1.000 (nessuna parte arriva al limite). Il combinatore
    vincolato, rifatto sulle stesse unità, ridà le accuratezze del file su
    tutte. Esito: sotto l'angolo utile (1.383 unità) la coppia supera la più
    forte in 117 unità con il vincolo e in 411 su 1.354 senza (altre 7 toccano
    il tetto, 22 scendono al livello del caso), ma senza vincolo le sta sotto
    in 455: sopra e sotto si equivalgono (test del segno, p = 0,14), e il
    rapporto mediano dS/dB resta 1,00 (quartili 0,98 e 1,01). Fra le unità che
    la superano, il rapporto mediano è 1,030 senza vincolo e 1,018 con. La
    formula senza vincolo prevede lì un guadagno mediano del 3,3 per cento, e
    la correlazione di rango fra guadagno previsto e misurato è 0,14. Il peso
    della rete debole è negativo in tutte e cinque le parti in 476 unità. Sopra
    l'angolo utile (1.232 unità) il vincolo quasi non conta: senza, 533 unità
    sopra la più forte e 364 sotto; con, 496 e 337; mediana 1,00 in entrambi.
  - *Il turing senza risposta in chat.* Motivo: la riga «da sola» di ChatBench
    mescola le prove con e senza una risposta registrata in chat, e quelle
    senza non erano mai state guardate da sole. Stessa procedura di
    `run_chatbench.py` (`compare`, quattro opzioni, 2.000
    ricampionamenti) sulle prove complementari. GPT-4o: 372 prove (124
    persone, 222 domande), accuratezza della persona 0,49, della macchina 0,81,
    della coppia 0,50, turing 1,03 [0,77; 1,44], dS/dM 0,44 [0,27; 0,60].
    Llama 3.1 8B: 432 prove (128 persone, 247 domande), 0,54, 0,53 e 0,57,
    turing 1,11 [0,92; 1,38]. Con le prove che hanno una risposta in chat
    (1.121 e 969) fanno le 1.493 e 1.401 della riga «da sola». Per la
    taratura (T) questi intervalli vanno letti come al 99 per cento circa.
    Aggiunta della stessa sera (quarta revisione): le prove senza
    risposta divise secondo l'annotazione di ChatBench su come la persona ha
    usato la chat (`user_effort`, assegnata da un modello linguistico), senza
    intervalli, in `by_effort`. GPT-4o: «tried» 264 prove (persona 0,49,
    macchina 0,79, coppia 0,50), «low effort» 92, «known» 16. Llama 3.1 8B:
    «tried» 352 (0,52, 0,50, 0,56), «low effort» 65, «known» 15. Descrive
    prove scelte dopo il trattamento: non dice perché la risposta non esce.
    Dopo una quinta revisione (stessa sera), in `check_3`: `by_kind` divide
    le prove secondo che cosa ha registrato la chat (`answer`, una lettera
    fra le opzioni; `out_of_options`, una risposta annotata che non è una
    lettera; `no_answer`), con accuratezze e cambi di risposta. GPT-4o:
    330 senza risposta (persona 0,50, macchina 0,84, coppia 0,50; cambia il
    29%, 8,2% verso il giusto e 8,2% verso lo sbagliato) e 42 fuori opzione;
    dove la chat ha risposto i passaggi al giusto sono otto volte quelli allo
    sbagliato. `in_chat_widened` conta come sbagliate le risposte fuori
    opzione: la macchina in chat va da 0,84 a 0,79 con Llama (da sola 0,71
    sulle stesse prove) e da 0,92 a 0,88 con GPT-4o (da sola 0,90).
    `low_effort_share`: 0,28 fra le prove senza risposta di GPT-4o, 0,07 su
    tutte.
  - *L'incertezza delle tendenze.* Motivo: le correlazioni di rango su dieci
    macchine erano riportate senza incertezza. p per permutazione esatto (le
    10! = 3.628.800 permutazioni dei ranghi), intervallo percentile
    ricampionando le macchine (10.000 volte), la correlazione togliendo una
    macchina alla volta; per αNLI anche ricampionando i casi con gli stessi
    pesi per tutte le macchine. Esito: visione −0,45, p bilaterale 0,19
    (unilaterale 0,096), intervallo [−0,95; 0,43]. ChatBench in risposta secca
    −0,73, p 0,020, intervallo [−0,99; −0,06]; senza gemma-4-26b −0,63, p
    0,076 (unilaterale 0,038). ChatBench con il ragionamento −0,02, p 0,97.
    αNLI −0,96 e −0,95, p 5·10⁻⁵ e 1·10⁻⁴, intervalli [−1,00; −0,79] e
    [−1,00; −0,76], ricampionando i casi [−0,99; −0,79] e [−0,99; −0,75].
    Con dieci punti gli intervalli sulle macchine sono grezzi. Per ChatBench e
    per la visione non si è ricampionato a livello di casi e persone:
    l'incertezza di ciascun φ (una ventina di gradi su ChatBench) entra solo
    attraverso la dispersione fra le macchine.
  - *La larghezza degli intervalli.* La taratura (T) diceva «il meccanismo non è
    stato isolato». Per il turing sul disegno di ChatBench la larghezza (1,3-1,7
    volte il necessario) è compatibile con il pigeonhole bootstrap di Owen (2007),
    che conta fino a tre volte la varianza residua, cioè fino a √3 in larghezza.
    Non spiega l'angolo (1,6-2,1 volte) né lo scenario F (2,4): lì il meccanismo
    resta non isolato. Nessuna prova nuova, è una lettura della letteratura.

## Deviazioni del 14 novembre 2026

Decise dopo tutti i risultati qui sopra, per rispondere a una domanda: la
lettura regge con macchine molto più forti di GPT-4o? Esplorative per intero.

- **Quattro modelli chiusi.** Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 e Fable
  5.2 rispondono da soli alle domande di ChatBench (396) e αNLI (1.532), con il
  prompt `letter` invariato, attraverso Claude Code headless, con impegno `low`
  (sui modelli più grandi il ragionamento non si spegne), una risposta per
  domanda, la lettera vincolata da uno schema di risposta. Solo verdetti: il
  combinatore a probabilità di H3 non si applica. Nessuna risposta persa.
- **Che cosa si calcola.** Con gli stimatori di H1 e H7: k, φ, l'angolo utile e
  il margine della coppia ideale sopra il più forte dei due,
  |S|/max(dH, dM) − 1, anche a 90 gradi. Su αNLI con le tre letture della
  chiave della prova 1 del 10 novembre (la chiave della storia, senza le storie
  in cui la maggioranza dei votanti la contraddice, la maggioranza come chiave),
  per i modelli chiusi e per quelli aperti allo stesso modo. `angle/frontier.py`,
  `results/frontier.json`.
- **Limiti noti prima di leggere i numeri.** Su ChatBench i modelli più forti
  non sbagliano quasi mai, e senza errori né d′ né φ esistono: l'accuratezza è
  fermata a 0,9999, quindi lì k è un minimo. Una risposta per domanda, a un solo
  livello di impegno, attraverso un prodotto in abbonamento e non l'API: la
  versione del modello è quella servita quel giorno.

## Deviazioni del 15 novembre 2026

Chieste da due revisioni dell'articolo, dopo tutti i risultati qui sopra.
Esplorative e descrittive: nessun intervallo è tarato, e le prove con una
risposta in chat sono una selezione. `angle/conversations.py`,
`results/conversations.json`.

- **Dove vanno i punti.** La distanza in accuratezza fra la macchina da sola e
  la coppia, divisa fra le prove in cui la chat ha dato una delle opzioni (la
  conversazione: chat meno macchina da sola; la decisione: coppia meno chat) e
  le altre. Una partizione delle prove, senza convenzioni per le chat mute. Una
  scomposizione in d′ su tutte le prove, con le chat mute contate come risposte
  a caso, è stata provata prima: dipende da quella convenzione (con quella
  opposta i segni di e ed f si rovesciano), ed è stata tolta.
- **Estrazione e filtro.** Tu = k·e·f sulle prove con una risposta, dalle
  sensibilità già in `results/chatbench-source.json`.
- **Seguire la chat.** Sulle prove in cui la lettera della chat è diversa dalla
  risposta da sola della persona: quanto la si segue quando ha ragione e quando
  ha torto, con intervalli che ricampionano le persone, e persone e domande
  insieme. Contro un osservatore lineare che aggiunge un bonus fisso
  all'opzione della chat, tarato sull'accuratezza della coppia: dà una
  differenza simile, quindi la differenza da sola non prova un gating.
- **Sempre e mai.** Quote di persone con almeno due disaccordi che hanno
  seguito sempre o mai, contro persone con le stesse probabilità.
- **Bravura.** Metà delle prove di ogni persona, scelte a sorte, classifica;
  l'altra metà misura. Mediana su 300 divisioni, terzo più basso e più alto con
  i pareggi. Ogni persona ha una sola materia, quindi anche nella sola
  matematica. Una prima divisione singola era stata letta nell'articolo ed è
  stata sostituita: era un estremo, e confusa con la materia.
- **Zero errori.** Per un modello senza errori su ChatBench il ricampionamento
  delle domande non muove l'accuratezza della macchina: il limite viene dalla
  regola del tre (un errore fino a 3/396), con la sensibilità della persona al
  suo valore e all'estremo alto del suo intervallo.

- **Più tardi lo stesso giorno, dopo una terza revisione.** Altre quattro
  analisi descrittive nello stesso modulo. *Gruppo contro persona*: la
  dispersione dell'accuratezza delle persone, tolto il rumore binomiale e
  approssimata con una distribuzione Beta, dà il rapporto fra le sensibilità
  medie delle singole persone con e senza la macchina, da mettere accanto al
  turing del gruppo (Macmillan e Kaplan, 1985); bootstrap sulle persone.
  *Partita sbagliata*: sulle conversazioni con una risposta, quanto la chat
  supera la macchina da sola quando la risposta da sola della persona era
  giusta e quando era sbagliata, e quanto spesso una chat sbagliata ripete la
  risposta sbagliata della persona. *Cento domande*: errori delle persone,
  della macchina e di tutte e due, contro errori indipendenti. *Contatto*: la
  scomposizione in quattro fattori in d′ su tutte le prove, con le due
  convenzioni per una chat muta.

- **Più tardi lo stesso giorno, dopo una revisione finale.** Due correzioni.
  *Lettere minuscole*: ChatBench registra tredici risposte della chat in
  minuscolo («c») e il confronto con la chiave le contava come sbagliate; la
  lettera ora si normalizza dove si leggono le conversazioni
  (`sources.chatbench_conversations`), e sono stati rieseguiti
  `run_chatbench.py`, la verifica 3 di `review.py` e `conversations.py`. Con
  GPT-4o cambiano tre prove, con Llama una; nessuna conclusione cambia. *La
  trappola della domanda*: se una chat sbagliata ripete la risposta sbagliata
  della persona si confronta non con un terzo ma con la quota delle altre
  persone che, sbagliando da sole quella domanda, avevano scelto la lettera
  della chat; intervalli con pesi di Poisson sulle persone, e su persone e
  domande.

- **Più tardi lo stesso giorno, dopo un'altra revisione.** Accordo e
  disaccordo (`conversations.py`, `agreement_and_disagreement`), descrittivo:
  sulle chat con una lettera, quanto spesso la chat sbaglia quando dà la stessa
  risposta della persona da sola e quando no (intervalli con pesi di Poisson
  sulle persone, e su persone e domande), chi ha ragione in un disaccordo, per
  materia e per la sicurezza dichiarata dalla persona; seguire la chat in ogni
  disaccordo contro ciò che ha fatto la coppia. Nel modello: i pesi della
  coppia ideale, (k − cos φ)/(1 − k cos φ); i pesi uguali («vince il più
  sicuro»); un osservatore che aggiunge all'opzione della chat il bonus del
  rapporto di verosimiglianza.

## L'avvocato del diavolo (scritto il 15 novembre 2026, prima di ogni corsa)

Scritto dopo i risultati dell'articolo e prima di qualunque chiamata a una
macchina nelle due condizioni nuove. Il codice (`angle/devil.py`,
`closed/devil_claude.py`, `gpu/devil_llm.py`) è stato scritto e provato prima su
prove sintetiche; questa sezione e il codice sono salvati e pubblicati prima
delle corse.

**La domanda.** Le conversazioni mostrano una trappola: una chat sbagliata dopo
una persona sbagliata ripete la risposta della persona più spesso di quanto
farebbero altre persone sbagliate sulla stessa domanda. Quando la macchina vede
la risposta della persona, cade nel suo errore? E chiederle prima gli argomenti
contro quella risposta toglie la trappola, e porta la coppia sopra la macchina
da sola?

**Dati.** ChatBench, condizione answer-first, fase 2: le 2.903 risposte date da
sole (329 persone, 396 domande, 1.187 sbagliate). Ogni coppia distinta (domanda,
risposta data da sola) è un item, 1.038 in tutto
(`data/items/chatbench-devil.jsonl`). L'unità di analisi è la prova (persona,
domanda): nelle condizioni A e B il verdetto della macchina dipende dalla
risposta della persona, quindi è assegnato prova per prova.

**Condizioni.** Una chiamata per item e per condizione.

- *0, da sola*: il prompt `letter` di `gpu/machines_llm.py`, senza la risposta
  della persona. Per Claude Haiku 5.5 la corsa già in
  `closed-results/chatbench__haiku-5.5__low.jsonl`; per Llama una corsa nuova.
- *A, informata*: lo stesso prompt con la riga `I think the answer is X.` prima
  della richiesta, X la lettera scelta dalla persona da sola.
- *B, avvocato del diavolo*: la riga `I think the answer is X.`, poi `First
  give, briefly, the strongest arguments against my answer. Then finish with a
  final line of the form: Answer: Y, where Y is the letter of the correct
  option (A to D).` Claude risponde con uno schema a due campi, prima gli
  argomenti e poi la lettera.

**Macchine.** Llama 3.1 8B Instruct, il modello aperto con cui hanno lavorato le
persone di ChatBench (pesi dalla copia senza accesso riservato
`unsloth/Llama-3.1-8B-Instruct`, uguali a quelli di Meta), con vLLM e
decodifica greedy, la lettera letta dalle probabilità vincolate come per gli
altri modelli aperti; in B prima si generano gli argomenti, fino a 700 token.
Claude Haiku 5.5 attraverso Claude Code in modalità non interattiva, effort
`low`, come per le altre corse Claude. Un item a cui una macchina non risponde
in una condizione esce da tutte e tre, per quella macchina.

**Misure**, per macchina.

1. L'accuratezza della macchina sulle 2.903 prove in 0, A e B, il suo d′ a
   quattro opzioni, e la resa di B, d′_B / d′_0. B è una coppia: la persona
   propone, la macchina attacca e decide.
2. L'accuratezza sulle prove in cui la persona da sola aveva sbagliato, e su
   quelle in cui aveva risposto giusto, per condizione.
3. La trappola: sulle prove in cui sbagliano tutte e due, la quota in cui la
   macchina dà la lettera della persona, per condizione.
4. Solo descrittivo: l'angolo fra gli errori della persona e quelli della
   macchina, per condizione, con lo stimatore dell'esperimento
   (`measure.compare`, macchina per prova). In A e B la macchina ha visto la
   risposta della persona: il modello gaussiano della coppia ideale non regge,
   e il margine non si calcola.

**Ipotesi e criteri.** Intervalli percentili al 95%, 2.000 repliche che
ricampionano insieme persone e domande (pesi di Poisson, come in `measure`).

- *H1, la trappola*: sulle prove in cui la persona aveva sbagliato, la
  macchina informata è meno accurata della macchina da sola. Passa se
  l'intervallo di acc_A − acc_0 sta sotto lo zero.
- *H2, l'avvocato del diavolo toglie la trappola*: sulle stesse prove,
  acc_B − acc_A sopra lo zero.
- *H3, la coppia sopra la macchina*: su tutte le prove, d′_B / d′_0 sopra uno.
- *Riportati comunque*, senza criterio: il costo sulle prove in cui la persona
  aveva risposto giusto, acc_B − acc_0; le quote della trappola; gli angoli.

Tre prove per macchina, due macchine, nessuna correzione per la molteplicità.

**Che cosa non può dire.** Nessuna persona legge gli argomenti: decide la
macchina. Un H3 positivo dice che, su queste domande, una coppia in cui la
risposta della persona è l'ingresso di una macchina che la contesta batte la
macchina da sola; non che lo farebbero persone che leggono gli argomenti. Una
chiamata per item: la variabilità della macchina fra una chiamata e l'altra non
è negli intervalli.

**Dove va nell'articolo.** Se H3 passa per almeno una macchina, nel corpo;
altrimenti nell'appendice di ciò che non è andato. In tutti e due i casi con i
numeri delle tre ipotesi per le due macchine.

**Deviazioni, lo stesso giorno, dopo le corse di Claude Haiku 5.5 e prima di
quella di Llama.** I risultati di Haiku erano già visti; nessuna delle due scelte
dipende da loro.

- *Llama a 8 bit.* I pesi bf16 (16 GB) avrebbero lasciato 2 GB liberi sul
  disco della macchina con la GPU, che è la macchina di lavoro di una persona.
  Llama gira con i pesi FP8 dello stesso modello
  (`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`, 9 GB, pesi e attivazioni a 8 bit).
- *Haiku da sola, di nuovo.* La condizione da sola veniva da una corsa del 14
  novembre, le altre due da oggi: una differenza fra due corse potrebbe passare
  per un effetto delle condizioni. Haiku da sola viene interrogata di nuovo oggi
  con lo stesso prompt (`closed-results/chatbench__haiku-5.5-rerun__low.jsonl`).
  È un controllo deciso dopo aver visto i risultati, e si riporta come tale: le
  prove del protocollo restano sulla corsa indicata sopra.

## Dopo la sesta revisione (15 novembre 2026)

Scritto dopo tutti i risultati qui sopra, e dopo una sesta revisione
dell'articolo. Non ha a che fare con l'avvocato del diavolo: quella sezione, il
suo codice e i suoi risultati qui non si toccano. Tutto ciò che segue è stato
deciso **dopo** i risultati: è esplorativo o descrittivo, senza ipotesi né
criteri, e un esito favorevole non vale come conferma. Codice in
`angle/verdict.py` e `angle/difficulty.py` (nessuna funzione esistente è
cambiata), numeri in `results/verdict-ceiling.json` e
`results/difficulty.json`, prove in `tests/test_verdict.py` e
`tests/test_difficulty.py`.

- **Il tetto con un verdetto.** Motivo: la coppia ideale dell'articolo mette in
  comune gli indizi graduati dei due, e il margine si misura contro di lei; chi
  lavora accanto a una macchina di solito ne riceve il verdetto, e una coppia
  che si scambia verdetti a quel tetto non arriva. Metodo: l'osservatore ottimo
  (massimo a posteriori) che tiene i propri indizi graduati e dell'altro vede
  solo l'opzione scelta, nel modello dell'articolo (quattro opzioni, un indizio
  gaussiano per opzione, il rumore dei due correlato opzione per opzione con ρ,
  come in `sdt._simulate_correct`). Dati gli indizi dell'osservatore e
  un'ipotesi sull'opzione vera, la probabilità di ogni verdetto è un integrale
  in una dimensione; l'osservatore risponde l'opzione con la probabilità a
  posteriori più alta. Gli indizi si estraggono (200.000 prove), il verdetto
  no: ogni prova è mediata sui quattro verdetti. Gli errori della coppia sono
  un rapporto con quelli del più forte dei due sulle stesse prove, ancorato al
  suo tasso d'errore esatto; dove l'osservatore è il più forte e non sbaglia
  quasi mai, i suoi indizi si estraggono con un rumore più largo e si pesano
  di conseguenza. Provato contro due forme chiuse: errori indipendenti con
  quattro opzioni (il verdetto vale un premio fisso ln(p/q)/d sulla sua
  opzione), ed errori correlati con due opzioni (un integrale in una
  dimensione). L'osservatore è esatto, quindi non è tenuto ai pesi non negativi
  come l'ideale dell'articolo: sotto l'angolo utile ciò che guadagna si riporta
  com'è.
  (a) Le righe della tabella del margine dell'articolo: la persona è il gruppo
  di ChatBench accanto a GPT-4o, che da solo risponde giusto 57,7 volte su
  cento (l'articolo stampa 58; con 0,58 esatto gli errori tolti verrebbero un
  punto più alti di quelli stampati), la macchina 1,2, 2, 3, 5 e 10 volte più
  sensibile, a 75 e a 90 gradi. Con gli indizi in comune il codice ridà la
  tabella com'è stampata.
  (b) Le due coppie di ChatBench, macchina interrogata da sola, tutte le prove,
  con le loro sensibilità e il loro angolo misurati, ai due estremi
  dell'intervallo dell'angolo e a 90 gradi.
  Esito, con il solo verdetto della macchina e la persona che tiene gli indizi.
  A 90 gradi il margine sopra la macchina è del 17,0 per cento con una macchina
  1,2 volte più sensibile (30,2 con gli indizi in comune), del 2,1 con una due
  volte più sensibile (11,8), dello 0,1 con una tre volte (5,4), e niente da
  cinque volte in su (2,0 e 0,5). A 75 gradi: 8,9 per cento (16,4), 0,1 (3,1),
  poi niente (0,3, 0, 0). In errori della macchina tolti, a 90 gradi: 17,7 per
  cento (30,3 con gli indizi in comune), 5,6 (28,2), 0,3 (27,2), niente (26,5 e
  25,7); a 75 gradi: 9,5 (17,0), 0,3 (8,0), niente (1,7, 0, 0). Il quarto degli
  errori della macchina che la coppia ideale toglie a 90 gradi qualunque sia k
  esiste solo se gli indizi sono in comune. Errori standard Monte Carlo: al più
  0,15 punti.
  Con il solo verdetto della persona e la macchina che tiene gli indizi, il
  caso del combinatore a verdetto di H3 e H5: margine a 90 gradi 19,4, 7,0,
  3,0, 1,1 e 0,2 per cento, a 75 gradi 11,0, 1,8, 0,1, 0,1 e 0,7; sotto
  l'angolo utile (cinque e dieci volte, a 75 gradi) l'osservatore esatto va una
  frazione di punto percentuale sopra il più forte, dove l'ideale dell'articolo
  si ferma a lui.
  ChatBench. Con GPT-4o (k = 2,16, 74,5 gradi): l'ideale con gli indizi in
  comune vale 1,020 volte la sensibilità della macchina, il tetto con il suo
  verdetto 1,000 (1,000 e 1,007 agli estremi dell'intervallo dell'angolo, 59 e
  86 gradi; 1,013 a 90), e le persone sono arrivate a 0,761 (0,673-0,849). Con
  Llama 3.1 8B (k = 1,18, 75,1 gradi): 1,173 con gli indizi in comune, 1,098
  con il suo verdetto (1,047 e 1,132 a 62 e 82 gradi; 1,180 a 90), e le persone
  sono arrivate a 1,189 (1,044-1,340), sopra tutti e due, perché la macchina
  interrogata da sola è più debole di quella che parlava con loro
  («Deviazioni», 9 novembre).
  È un modello, non una misura, e i suoi errori sono errori Monte Carlo, non
  intervalli di campionamento. Una macchina che dà anche le sue ragioni, o una
  sicurezza graduata, dà più di un verdetto: il tetto vincola una coppia che si
  scambia verdetti, non le persone di ChatBench, che leggevano una
  conversazione.

- **La vicinanza al netto della difficoltà, su αNLI.** Motivo: nell'angolo
  finiscono insieme due cose, quanto due menti si somigliano e quanto i casi
  sono difficili per tutti. L'articolo lo dice, dice che i dati per separarle
  ci sono, e la revisione ha chiesto di usarli. Per la coppia non cambia
  niente: la difficoltà condivisa conta in ρ («Modello e definizioni») e il
  margine tiene tutta la correlazione. Cambia quanta parte dell'angolo si può
  leggere come somiglianza.
  *Chi è la persona.* Su αNLI la «persona» è un voto dei cento voti anonimi su
  una storia (`measure.compare_votes`): non ci sono persone da seguire, solo
  quanti hanno votato per la chiave. La correlazione fra persona e macchina è
  quindi, per costruzione, quanto la macchina sbaglia dove sbagliano in molti,
  e la difficoltà per le persone ne è tutto il lato umano. Una difficoltà
  stimata sulle persone stesse si porterebbe via tutto. La si mostra solo per
  dire perché non è l'analisi fatta qui: con metà dei voti di ogni storia a
  fare gli strati e l'altra metà come persona, l'angolo con Fable 5.2 passa da
  47,0 gradi a 62,7 con tre fasce (58,1-66,9), 82,9 con dieci (79,3-86,2) e
  86,6 con venticinque (83,1-89,9). Con tre fasce ciò che resta è difficoltà
  dentro la fascia: lì un angolo sotto i 90 gradi non prova niente.
  *Metodo.* La difficoltà si stima su una terza popolazione. Persona e
  macchina: le dieci macchine aperte nelle due modalità, venti risposte per
  storia (per una macchina aperta le altre nove, diciotto risposte); il
  punteggio di una storia è quante sono giuste. Macchina e macchina: i voti
  per la chiave. Strati: i livelli consecutivi del punteggio, uniti a partire
  dal lato difficile finché ciascuno ha almeno 60 storie (da 9 a 12 strati sul
  punteggio delle macchine; 18 sui voti, 15 senza le storie contestate):
  abbastanza fini da lasciare poca difficoltà dentro uno strato, abbastanza
  larghi da avere errori da contare. Con 30 e con 120 storie le correlazioni
  si spostano al più di 0,04; con la sola risposta secca come terza popolazione
  (dieci risposte) quelle fra persona e macchina sono più alte fino a 0,08,
  perché un punteggio più grosso toglie meno. Stimatore: una sola correlazione
  latente per tutti gli strati, ogni strato con le sue due soglie prese dai
  suoi margini, al massimo della somma delle log-verosimiglianze delle tabelle
  2×2; con uno strato solo è la tetracorica dell'articolo. Uno strato in cui
  uno dei due non sbaglia mai non porta informazione ed esce. Accanto, il
  rapporto di odds comune di Mantel-Haenszel, che non ha bisogno di soglie.
  Intervalli: 2.000 ricampionamenti delle storie (pesi di Poisson), strati
  fissi; 250 per la mediana sulle 180 coppie di macchine aperte. Prova: su
  storie simulate con una correlazione nota dentro ogni livello di difficoltà
  lo stimatore la ridà (0,29 per una vera di 0,30, dove la correlazione su
  tutte le storie insieme è 0,64; alla taglia di αNLI con diciotto strati e
  una macchina che risponde giusto 88 volte su cento, una media di 0,30 su
  dodici corpora).
  *Esito, persona e macchina*, al netto della difficoltà per le macchine
  aperte. L'angolo si apre e resta sotto i 90 gradi con tutto l'intervallo per
  i quattro modelli Claude: Fable 5.2 da 47,0 (44,4-49,9) a 58,6 (55,4-61,8),
  Opus 5.5 da 47,9 a 59,9 (56,8-63,1), Sonnet 5.5 da 49,8 a 64,9 (61,1-68,5),
  Haiku 5.5 da 55,3 a 74,7 (70,6-78,5). Senza le 157 storie contestate: Fable
  da 67,5 a 72,2 (68,6-76,2), Opus da 70,2 a 74,1 (70,5-77,9), Sonnet da 68,8
  a 76,4 (72,9-79,7), Haiku da 71,3 a 81,6 (78,6-84,8). Della correlazione fra
  le persone e Fable restano tre quarti (da 0,68 a 0,52, intervallo 0,47-0,57;
  76 per cento, 72-81), di quella con Haiku meno della metà (da 0,57 a 0,26,
  0,20-0,33; 46 per cento, 37-55). Le macchine aperte più forti, al netto
  delle altre nove, in risposta secca: Gemma 4 26B da 55,5 a 72,7 (69,1-76,5),
  Qwen3.8 27B da 52,9 a 70,3 (66,2-74,2). Per le sette macchine aperte più
  deboli l'angolo al netto delle altre sta fra 84,5 e 94,4 gradi nelle due
  modalità, e in undici casi su quattordici l'intervallo comprende i 90 gradi
  o sta sopra: ciò che hanno in comune con le persone è ciò che è difficile
  per le macchine aperte.
  *Esito, macchina e macchina*, al netto della difficoltà per le persone. Opus
  5.5 e Fable 5.2 restano quasi la stessa macchina: da 0,95 (0,92-0,97) a 0,83
  (0,75-0,91). Sonnet e Fable da 0,88 a 0,59 (0,45-0,72). Haiku e Fable da
  0,72 a 0,24 (0,05-0,41), Gemma 4 26B e Fable da 0,71 a 0,24 (0,05-0,42),
  Gemma e Haiku da 0,77 a 0,52 (0,41-0,63). La mediana sulle 180 coppie di
  macchine aperte passa da 0,63 (0,60-0,66) a 0,45 (0,43-0,49). I rapporti di
  odds dicono lo stesso: Opus e Fable da 156 a 20 (11-41), Haiku e Fable da 16
  a 1,9 (1,1-3,1), Gemma e Fable da 15 a 2,0 (1,2-3,3). Senza le storie
  contestate: Opus e Fable da 0,89 a 0,85 (0,73-0,95), Haiku e Fable da 0,44 a
  0,16 (da −0,13 a 0,39), Gemma e Fable da 0,56 a 0,28 (0,01-0,51). Con le
  macchine più forti contano solo gli strati difficili: Fable non sbaglia mai
  in nove dei diciotto strati, e un confronto con lei poggia su 573 storie.
  *Che cosa non può dire.* Il punteggio è un conto di venti risposte, o di
  cento voti: la difficoltà che lascia dentro uno strato resta nella
  correlazione, che è un limite superiore della somiglianza. E le macchine
  aperte non sono giudici neutrali: ciò che hanno in comune con la macchina
  misurata se ne va con la difficoltà, e tanto più quanto quella macchina
  somiglia a loro. Haiku e le macchine aperte deboli sono vicine alla terza
  popolazione, Fable no: gli angoli al netto di macchine diverse non sono una
  tendenza da leggere. Un angolo «puro» solo non esiste, dipende da chi misura
  la difficoltà. Qui niente è tarato.

- **Le storie contestate su metà dei voti.** Motivo: l'articolo toglie le
  storie di cui la maggioranza dei votanti contraddice la chiave guardando gli
  stessi voti che poi misurano l'angolo, e lo dichiara. Metodo: metà dei cento
  voti di ogni storia, estratti a sorte, dicono se è contestata; l'altra metà
  è la persona; stesso passo di `compare_votes`. Punto: media su 200
  divisioni. Intervallo: 2.000 ripetizioni, ciascuna con una divisione nuova e
  le storie ricampionate. Esito: l'angolo senza le storie contestate è 65,9
  gradi per Fable 5.2 (62,3-70,0) contro 67,5 con gli stessi voti, 68,3 per
  Opus 5.5 (64,2-72,8) contro 70,2, 67,8 per Sonnet 5.5 (64,1-71,9) contro
  68,8, 70,7 per Haiku 5.5 (67,5-74,1) contro 71,3, 69,0 per Gemma 4 26B in
  risposta secca (65,6-72,0) contro 69,4. Il margine della coppia ideale: 3,9
  per cento per Fable (2,3-5,7) contro 4,5, 6,4 per Opus (4,2-8,9) contro 7,2,
  8,7 per Sonnet (6,0-11,3) contro 9,2, 20,0 per Haiku (16,0-23,3) contro
  20,7. La scelta fatta sugli stessi voti vale uno o due gradi, e la lettura
  non cambia. L'altra metà di quella cautela resta: togliere le storie più
  difficili per le persone ne restringe la variabilità, quali che siano i voti
  che le scelgono.

- **Le persone di silicio: il metodo, e il braccio con la macchina sana.**
  Nessun conto nuovo: i numeri stavano già in `results/silicon.json` (300
  ricampionamenti) e l'articolo non descriveva il metodo. Metodo
  (`gpu/silicon.py`): CIFAR-10; venti piccole reti convoluzionali per braccio;
  tre tappe, con 4.000, 12.000 e 36.000 immagini di addestramento viste in
  tutto, sei epoche ciascuna; la prima tappa da sole in ogni braccio. Dalla
  seconda tappa il controllo minimizza l'entropia incrociata con l'etichetta
  vera, i due bracci assistiti metà di quella più metà della divergenza di
  Kullback-Leibler dall'uscita della macchina (peso 0,5). La macchina è una
  ResNet-18 addestrata sull'insieme di addestramento e sulle 8.000 immagini di
  test fuori dall'esame, poi congelata. La sua copia difettosa, sulle immagini
  che chiama gatto o cane e le cui caratteristiche superano una soglia lungo
  una direzione casuale fissa (sette su dieci), mette l'altra classe due logit
  sopra. L'esame sono 2.000 immagini di test che né le reti né la macchina
  hanno visto in addestramento (la soglia del difetto è fissata su di loro);
  su 277 il difetto cambia la risposta della macchina. Le reti vedono dunque
  sempre l'etichetta vera, e il parere della macchina è metà di ciò da cui
  imparano: che il braccio con il difetto penda verso di lui è il verso che la
  funzione di costo impone. La misura no: dove la macchina aveva ragione prima
  del difetto, il bersaglio a cui la funzione di costo mira ha ancora la classe
  vera in testa (almeno una metà, contro al più 0,44 per la classe del
  difetto).
  I numeri. Sui 277 casi, la risposta sbagliata del difetto: 11,0 per cento
  nel controllo, 11,2 con la macchina sana, 23,2 con quella difettosa;
  difettosa meno controllo 12,1 punti (8,7-15,8), sana meno controllo 0,2 (da
  −1,0 a 1,5); risposte giuste lì 73,5, 73,9 e 61,1 per cento. Il braccio con
  la macchina sana contro il controllo, all'ultima tappa: correlazione delle
  risposte giuste con la macchina 0,669 contro 0,670 (0,407 e 0,405 alla
  prima), accuratezza 83,6 contro 83,3 per cento; differenza nelle differenze
  di A, senza condizionare alla classe, −0,005 (da −0,041 a 0,032), dentro la
  classe −0,016 (da −0,141 a 0,071). Imparare accanto a una macchina sana non
  ha avvicinato le reti a lei più che imparare da sole. Il braccio con il
  difetto contro la macchina difettosa: correlazione 0,539, contro 0,413 per
  il controllo misurato contro la stessa macchina (0,298 e 0,278 alla prima
  tappa); differenza nelle differenze di A senza condizionare alla classe
  0,106 (0,065-0,152), dentro la classe, come prescriveva il protocollo, 0,019
  (da −0,059 a 0,094). Le correlazioni semplici non hanno un intervallo: le
  risposte delle reti non stanno in questo repository.

## L'avvocato del diavolo: esito (15 novembre 2026)

Risultati in `results/devil.json`; le risposte in `devil-results/` e in
`closed-results/chatbench__haiku-5.5-rerun__low.jsonl`. Intervalli come scritti
prima (2.000 repliche, pesi di Poisson su persone e domande).

- **Claude Haiku 5.5.** H1 non passa: informata della risposta sbagliata della
  persona, la macchina è *più* accurata di 1,3 punti che da sola (da −1,5 a
  +4,3). H2 non passa: +1,5 (da −0,9 a +4,2). H3 passa: d′_B/d′_0 = 1,18
  (1,07–1,31); con Haiku da sola interrogata di nuovo lo stesso giorno, 1,13
  (1,03–1,26). Informata, senza la richiesta (non era un'ipotesi): 1,14
  (1,03–1,27), 1,10 (1,00–1,22) con la seconda corsa. Accuratezza: da sola 93,5
  (seconda corsa 94,5), informata 96,5, avvocato del diavolo 97,1. Gettoni in
  uscita per risposta: circa 170, 220 e 860. Fra le due corse identiche di
  Haiku da sola cambiano esito 28 domande su 396.
- **Llama 3.1 8B (FP8), come scritto prima.** H1 passa: −34,5 punti (da −41,6
  a −27,8). H2 passa: +52,0 (da +45,3 a +58,3). H3 passa: 1,41 (1,09–1,89).
  Accuratezza: da sola 47,0, informata 61,7, avvocato del diavolo 56,7. Ma da
  sola e informata rispondevano con la sola lettera, l'avvocato del diavolo
  dopo aver ragionato.
- **Llama, i controlli con il ragionamento (aggiunti dopo i suoi risultati,
  prima di farli girare).** Tutte e tre le condizioni ragionano: da sola 72,0,
  informata 72,3, avvocato del diavolo 56,7. Il confronto di H1: −18,5 (da
  −24,8 a −12,2). Quello di H2: +10,0 (da +3,6 a +16,5). Quello di H3: 0,67
  (0,57–0,77), la coppia dell'avvocato del diavolo sotto la macchina da sola.
  Dove la persona aveva ragione l'avvocato del diavolo scende da 87,5
  (informata) a 54,1.
- **La trappola** (sbagliano tutte e due, la macchina dà la lettera della
  persona): Llama con la sola lettera 41 → 98 → 44 per cento (da sola,
  informata, avvocato del diavolo); Llama ragionando 48 → 85 → 44; Haiku 48 →
  46 → 50, su pochi errori.
- **Gli angoli**, descrittivi (ρ fra gli errori della persona e quelli della
  macchina): Llama con la sola lettera 0,15, 0,99 (al limite della conversione)
  e −0,10 (da −0,36 a 0,08); Llama ragionando, informata, 0,74; Haiku −0,02,
  0,26, 0,11.
- **Dove va.** H3 è passata per una macchina, quindi per la regola scritta
  sopra l'esperimento va nel corpo dell'articolo, con l'esito di Llama come era
  scritto e i controlli nella sua appendice.
