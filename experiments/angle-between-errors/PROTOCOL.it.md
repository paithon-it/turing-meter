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
