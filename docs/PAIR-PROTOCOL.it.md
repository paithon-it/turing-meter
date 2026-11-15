# Protocollo d'uso: misurare una coppia persona-macchina prima di fidarsi

> [English](PAIR-PROTOCOL.md) · **Italiano**. L'originale è questo; la versione inglese lo traduce.

Bozza 0.1, 15 novembre 2026. Accompagna l'articolo «Intelligenza Apparente» e usa le sue grandezze. L'articolo chiama «misura di partenza» ciò che qui si fa, perché usa «taratura» per la taratura dello strumento stesso (la sua appendice C); in queste pagine «taratura» è la misura di partenza di chi usa il modello.
Non è stata provata con persone vere: i conti sono simulazioni fatte con gli stimatori del repository
dell'esperimento, più due controlli sui dati di ChatBench. Ogni cifra dice da dove viene: **[articolo]**
è una cifra dell'articolo, **[simulazione]** un conto di queste pagine dentro il modello, **[dati]** un
conto di queste pagine sui dati pubblici di ChatBench.

## In breve

- **La domanda.** Voglio usare un modello, per esempio GPT-4o, e non so quanto vale rispetto a me.
  Serve una prova iniziale? Sì: *k*, quante volte la macchina vale la persona, non lo pubblica nessuno,
  perché dipende dalla persona e dal compito.
- **Che cosa si misura bene.** *k*. Con 100 casi cade fra 1,6 e 3,1 quando vale 2,2; con 200 fra 1,75 e
  2,8 [simulazione]. Basta a distinguere una macchina molto più forte da una alla pari.
- **Che cosa non si misura con cento casi.** L'angolo fra gli errori: con 100 casi cade fra 47 e 105
  gradi quando vale 75. Vive sugli errori della macchina: per conoscerlo entro dieci gradi ne servono
  fra 50 e 80, cioè quasi 700 casi con una macchina all'88 per cento e più di 2.000 con una al 98
  [simulazione].
- **La regola d'uso** si regge quindi su *k* e sul tetto del margine, non sul confronto fra l'angolo
  e l'angolo utile, che con cento casi non si sa da che parte stia.
- **La prova della coppia** si fa sugli stessi casi della taratura, non su casi nuovi: 200 casi dicono
  se la coppia sta sotto la macchina quando la resa è 0,76 come con GPT-4o.
- **La ripetizione** è asimmetrica. La macchina si rimisura a ogni versione, senza costo. Una persona
  sola, con cento casi per volta, vede un proprio calo solo se è un crollo; un'organizzazione lo vede.
- **Le insidie più gravi**: chiavi sbagliate quando la macchina è forte, casi già visti dal modello,
  un *k* unico dove i tipi di caso divergono, e la macchina della chat che non è quella della taratura.

## 1. A che cosa serve, e a che cosa no

Serve a chi prende decisioni che hanno una risposta giusta verificabile (una diagnosi poi confermata,
una classificazione, un controllo con un esito noto) e vuole sapere come dividere il lavoro con un
modello: seguirlo, controllarlo, o decidere insieme.

Non serve per il lavoro che non ha una chiave: scrivere, progettare, scegliere quale domanda farsi.
Lo strumento dell'articolo misura due tecniche sulla stessa domanda, e chi ha scelto la domanda gli
resta fuori [articolo]. Non misura il tempo, se non lo si cronometra a parte. E non dice che cosa
succederà fra sei mesi: vale per quei casi, quella versione del modello, quel modo di interrogarlo.

## 2. Quattro cose da sapere prima

**1. *k* è tuo, e cambia con il compito.** Nel gruppo di ChatBench che ha lavorato con GPT-4o le
accuratezze vere delle persone hanno una deviazione standard di 15 punti intorno al 58 per cento
[dati, dal file `conversations.json` del repository]. Chi sta una deviazione standard sotto la media
ha davanti una macchina che vale 3,9 volte lui; chi sta una sopra, 1,5 volte. E per le stesse persone
GPT-4o vale 2,33 volte in matematica, 3,18 in fisica e 1,08 in etica [articolo]. Un *k* medio su
tutto non descrive nessuno.

**2. La sensibilità della macchina non è una proprietà del modello.** È del modello su quei casi, con
quella richiesta. Chi produce il modello può pubblicare come risponde a un test aperto; quanto vale
sui tuoi casi lo misuri tu.

**3. L'angolo sta in un conto solo.** Dei casi che la macchina sbaglia, quanti ne sbagli anche tu?
Su ChatBench le persone sbagliano il 57 per cento delle domande che GPT-4o sbaglia, e il 40 per cento
di quelle che fa giuste [dati]. Se le due quote fossero uguali gli errori sarebbero indipendenti (90
gradi); più si staccano, più l'angolo si chiude. Per questo la precisione dell'angolo dipende da
quanti errori fa la macchina, non da quanti casi hai.

**4. La taratura misura la macchina da sola, non la chat.** In ChatBench Llama 3.1 8B rispondeva
giusto nell'85 per cento delle conversazioni da cui usciva una risposta, e nel 73 interrogata da sola
sulle stesse; con GPT-4o, in più di un quinto delle prove dalla chat non è uscita nessuna risposta
[articolo]. Per sapere come va la coppia bisogna provare la coppia.

## 3. Il disegno

### 3.1 I casi

- **Tipi di caso.** Si elencano prima i tipi di lavoro per cui il modello verrà usato. Ogni tipo ha
  la sua taratura e la sua regola.
- **Da dove vengono.** Dal lavoro vero, presi a sorte: non i casi interessanti, non quelli facili da
  trovare. Mai da test pubblici, e possibilmente successivi alla data in cui il modello ha smesso di
  imparare.
- **La chiave.** La risposta giusta va fissata prima che qualcuno risponda, da una fonte che non è né
  la persona né la macchina (un esito poi accertato, un secondo esperto). Un decimo delle chiavi, a
  sorte, va fatto rivedere da un secondo lettore che non sa chi ha risposto che cosa: la quota di
  chiavi che non reggono è un numero del referto. Dopo la prova si rivedono anche tutte le chiavi dei
  casi che la macchina ha sbagliato. Così si correggono solo gli errori a suo sfavore: quanto resta
  dall'altra parte lo dice la quota trovata nel decimo a sorte.
- **La forma.** Quattro opzioni, se il lavoro lo consente (paragrafo 3.5).
- **Quanti, e come dividerli.** Per una persona: 300 casi per tipo, divisi a sorte in tre moduli da
  100. Il modulo A serve alla taratura, il B alla prova della regola, il C resta da parte per la
  ripetizione. Dividere a sorte un solo mucchio rende i moduli difficili uguale, in media: è ciò
  che permette di confrontare la persona di oggi con quella di fra un anno. Il modulo C non va dato
  alla macchina finché non serve.
- **Ordine.** I casi si presentano in ordine casuale, diverso per la persona e per la macchina.

### 3.2 La sessione

1. **La persona da sola.** Risponde a tutti i casi del modulo senza vedere la chiave né la macchina.
   Per ogni caso segna la risposta, la sicurezza su tre livelli e il tempo.
2. **La macchina da sola.** Gli stessi casi, uno per chiamata, senza memoria fra un caso e l'altro,
   con una richiesta fissa. Si annotano: nome e data della versione (non il nome generico, che può
   passare a una versione nuova senza avviso), testo della richiesta, temperatura, strumenti accesi.
   Una chiamata per caso basta e non falsa le stime; tre o cinque dicono in più dove la macchina
   cambia risposta (paragrafo 7).
3. **La persona con la macchina.** Sugli stessi casi, in un secondo giro e in ordine diverso, la
   persona usa la chat come la userebbe davvero. Si annotano la risposta uscita dalla chat (o che non
   è uscita), la risposta finale e il tempo. Fra il primo giro e il secondo non si mostra nessuna
   chiave.
4. **Solo alla fine** si aprono chiavi e risultati. Da quel momento i casi del modulo, per quella
   persona, sono spesi.

Il terzo passo si fa sugli stessi casi, e non su casi nuovi, per tre ragioni: turing e resa vogliono le tre risposte sugli stessi
casi; è il disegno di ChatBench, da cui vengono le cifre con cui confrontarsi; e dimezza i casi. Il
prezzo è che la risposta con la macchina è un secondo tentativo. Nei dati pesa poco: dove la chat è
rimasta muta le persone restano al 50 per cento, e l'ordine delle mosse non cambia l'esito in modo
visibile [articolo]. I casi nuovi servono dopo, per provare la regola (paragrafo 5).

### 3.3 Quanti casi: i conti

**Come sono fatti.** Per ogni situazione (sensibilità vere, angolo vero, quattro opzioni) si simulano
4.000 tarature di una persona sola con il processo del repository, e su ciascuna girano gli stimatori
dell'articolo. Le tabelle danno l'intervallo in cui cade il 95 per cento delle stime: ciò che una
taratura sola può restituire. Controllo sui dati veri: estraendo da ChatBench tarature di 100 casi,
*k* e angolo oscillano quanto dice il modello (deviazione standard del logaritmo di *k* 0,171 contro
0,170; dell'angolo 14,5 gradi contro 14,8) [dati].

**Persona al 58 per cento, macchina all'88, come GPT-4o su ChatBench.** Vero: *k* 2,16, angolo 74,5
gradi, angolo utile 62,5, margine 2 per cento.

| Casi | Errori della macchina | *k* | Angolo | Margine misurato | Zona A decisa | «Angolo sopra l'utile» deciso |
|---|---|---|---|---|---|---|
| 50 | 6 | 1,5-3,8 | 13-127 | 0-57% | 45% | 8% |
| 100 | 12 | 1,6-3,1 | 47-105 | 0-25% | 72% | 14% |
| 200 | 23 | 1,75-2,8 | 56-95 | 0-14% | 94% | 24% |
| 400 | 47 | 1,9-2,6 | 62-88 | 0-9% | 100% | 44% |
| 800 | 93 | 1,95-2,4 | 66-83 | 0-6% | 100% | 72% |
| 1.600 | 187 | 2,0-2,3 | 68-81 | 0-5% | 100% | 93% |

«Zona A decisa» è la quota di tarature in cui tutto l'intervallo di *k* sta sopra 1,5 (paragrafo 4).
L'ultima colonna è la quota in cui l'intervallo permette di dire che l'angolo supera l'angolo utile,
cioè che la persona aggiunge qualcosa almeno in linea di principio: con cento casi succede una volta
su sette, e il margine in gioco è il 2 per cento.

**Alla pari, come GPT-4o sulle domande di etica** (71 e 74,5 per cento). Vero: *k* 1,08, angolo 70
gradi, margine 17 per cento.

| Casi | Errori della macchina | *k* | Angolo | Margine misurato | «*k* comprende 1» | «Margine sopra zero» deciso |
|---|---|---|---|---|---|---|
| 50 | 13 | 0,74-1,62 | 41-107 | 0-57% | 94% | 25% |
| 100 | 26 | 0,82-1,43 | 50-93 | 3-40% | 92% | 70% |
| 200 | 51 | 0,90-1,30 | 55-86 | 6-32% | 86% | 96% |
| 400 | 102 | 0,94-1,24 | 59-81 | 9-27% | 79% | 100% |

**Le situazioni in mezzo e le macchine più forti.**

| Situazione vera | Con 100 casi | Con 200 casi |
|---|---|---|
| *k* 1,18 (Llama 3.1 8B: 60,5 e 67%) | *k* 0,85-1,68; non si distingue da 1 nell'87% | *k* 0,94-1,50; non si distingue nel 74% |
| *k* 1,5 (65 e 82%) | *k* 1,13-2,05; «macchina più forte» deciso nel 79% | *k* 1,22-1,83; deciso nel 99% |
| *k* 2,5 (macchina al 93%) | *k* 1,85-3,8; zona A nel 93%; angolo 54-127 | *k* 2,0-3,3; zona A nel 100%; angolo 67-119 |
| *k* 3,2 (macchina al 98%) | *k* 2,3-6,1; zona A nel 97% (intervallo esatto); angolo illeggibile | *k* 2,6-4,6; zona A sempre; angolo illeggibile |

**Che cosa si può dire, quindi.**

- **Con 50 casi**: chi è più forte, se la distanza è grande. Nient'altro.
- **Con 100 casi**: *k* entro un terzo circa. Si distingue una macchina che vale almeno due volte e
  mezza la persona (zona A nel 93 per cento delle tarature) da una alla pari (l'intervallo comprende 1
  nel 92). Con *k* intorno a 2 la zona A esce sette volte su dieci. L'angolo non si legge.
- **Con 200 casi**: *k* entro un quarto. La zona A esce nel 94 per cento dei casi con *k* 2,2. Alla
  pari si può dire che la coppia ideale ha un margine sopra zero. La resa di una coppia come quella di
  ChatBench si distingue da uno. L'angolo ha ancora un intervallo di quaranta gradi.
- **Con 400 casi**: l'angolo entro 13 gradi con una macchina all'88 per cento.

**La regola pratica per l'angolo.** La sua deviazione standard vale circa 45 gradi diviso la radice
degli errori della macchina (fra 35 e 60 secondo le accuratezze), e l'intervallo al 95 per cento è il
doppio per parte.

| Errori della macchina | Intervallo dell'angolo, per parte | Casi con macchina all'80% | all'88% | al 93% | al 98% |
|---|---|---|---|---|---|
| 10 | circa 30, o più | 50 | 83 | 143 | 500 |
| 20 | 16-23 | 100 | 166 | 286 | 1.000 |
| 40 | 11-16 | 200 | 333 | 572 | 2.000 |
| 80 | 7-11 | 400 | 666 | 1.143 | 4.000 |
| 160 | 5-8 | 800 | 1.332 | 2.287 | 8.000 |

Sotto i 10 errori l'angolo non va riportato; fra 10 e 40 è un'indicazione.

**Gli intervalli.** Per una persona sola si ricampionano i casi. In simulazione questo intervallo
contiene il vero fra 92 e 97 volte su cento, per *k*, angolo e margine, quando i casi sono almeno
100 e la macchina ha almeno una dozzina di errori. Con pochi errori non regge: con una macchina al 98
per cento e 100 casi contiene il *k* vero 84 volte su cento, e l'angolo 40. Per *k*, sotto i cinque
errori della macchina, si usano gli intervalli esatti delle due accuratezze, più larghi e sempre
validi. Non sono gli intervalli dell'articolo, che ricampionano persone e casi insieme e sono da 1,3
a 2,1 volte più larghi del necessario [articolo].

### 3.4 Quando la macchina sbaglia pochissimo

Con una macchina al 98 per cento, su 100 casi si aspettano due errori, e una taratura su otto non ne
vede nessuno [simulazione]. Tre cose restano possibili.

1. **Un minimo per *k*.** Zero errori su *n* casi sono compatibili con un errore vero fino a 3/*n*:
   è la regola del tre, la stessa che l'articolo usa per Fable 5.2. Con 100 casi e nessun errore,
   una persona intorno al 58 per cento ha un *k* di almeno 1,9 [simulazione, `check_pair_tool.py`].
2. **Un tetto per il margine.** Il margine a 90 gradi scende quando *k* sale: all'estremo basso
   dell'intervallo di *k* si ha il margine più largo che una coppia ideale possa promettere con
   errori indipendenti. Con 100 casi e una macchina al 98 per cento quel tetto sta sotto il 20 per
   cento nel 97 per cento delle tarature, e sotto il 15 nel 73. È la costruzione dell'articolo per i
   modelli Claude (margine al più 11, 6,7, 6,2 e 5 per cento). Il limite è lo stesso: oltre i 90
   gradi, se gli errori si compensano, il margine supera il tetto, e senza errori della macchina non
   si può escluderlo. Nelle misure dell'articolo l'angolo mediano sta fra 59 e 81 gradi su tutte le
   fonti.
3. **Più errori senza più ore**, con il disegno a due fasi (paragrafo 3.7).

Ciò che non va fatto è scegliere i casi perché la macchina li sbaglia, senza pesarli: si misura un
altro *k*. E con macchine così forti le chiavi diventano il problema principale (paragrafo 7).

### 3.5 Scelta multipla, sì/no, risposte aperte

**Quattro opzioni** è la forma più efficiente. A parità di sensibilità, con casi sì/no *k* è più
incerto: con 100 casi cade fra 1,3 e 4,1 invece che fra 1,6 e 3,1, e ne servono circa 250 per la
stessa precisione [simulazione]. La ragione è che il 58 per cento su quattro opzioni dista dal caso 33
punti, mentre la stessa sensibilità sul sì/no dà il 70 per cento, che ne dista 20.

**Sì/no** ha due attenzioni in più. Le classi vanno bilanciate, e i conti vanno fatti dentro ciascuna
classe (i sì e i no della chiave). Se persona e macchina pendono tutte e due verso la stessa risposta
sbagliano insieme su una classe, e un conto unico legge quell'inclinazione come somiglianza: con
un'inclinazione comune moderata l'angolo esce 68 gradi invece di 75 [simulazione]. E l'allarme vale
meno: con due opzioni un disaccordo vuol dire solo che uno dei due ha torto, mentre con quattro due
errori cadono di rado sulla stessa opzione. Nel modello, ai valori di GPT-4o, la macchina sbaglia nel
2,6 per cento degli accordi e nel 12,5 dei disaccordi con due opzioni, nel 6 e nel 19 con quattro.

**Risposte aperte.** La sensibilità si ricava dall'accuratezza e dal numero di alternative, e una
risposta aperta non ne ha uno. Le stesse accuratezze, 58 e 88 per cento, danno *k* 5,8 se si contano
due alternative, 2,1 con quattro, 1,7 con dieci, 1,3 con mille [simulazione]: *k*, angolo utile e
margine dipendono da una convenzione che nessuno può fissare. Per il lavoro a risposta aperta restano
i conti che non la chiedono: le accuratezze, la quota di errori in comune, l'allarme (accordo vuol
dire stessa risposta nella sostanza, giudicata da chi non sa chi l'ha data), e la distanza in punti
fra la macchina da sola e la coppia, come nella tabella «Dove vanno i punti» dell'articolo. Oppure si
trasformano i casi in scelte chiuse, sapendo che riconoscere la risposta giusta fra quattro non è lo
stesso mestiere che produrla.

### 3.6 Una persona o un gruppo

Un gruppo che mette in comune i casi misura il *k* del gruppo e l'angolo del gruppo. Simulazione con
lo strumento dell'articolo, persone al 58 per cento in media con la dispersione di ChatBench, macchina
all'88, due persone fra loro più simili di quanto lo siano alla macchina (66 gradi contro 75
[articolo]). Sono 300 tarature per riga, e per questo la prima riga si scosta di poco dalla tabella
del paragrafo 3.3:

| Disegno | Risposte umane | Casi con la chiave | *k* del gruppo | Angolo |
|---|---|---|---|---|
| 1 persona, 100 casi | 100 | 100 | 1,6-3,1 | 52-105 |
| 1 persona, 500 casi | 500 | 500 | 1,9-2,5 | 64-87 |
| 5 persone, gli stessi 100 casi | 500 | 100 | 1,6-3,5 | 59-93 |
| 5 persone, 100 casi diversi a testa | 500 | 500 | 1,6-3,4 | 65-86 |
| 20 persone, gli stessi 100 casi | 2.000 | 100 | 1,8-3,0 | 63-87 |
| 20 persone, 25 casi diversi a testa | 500 | 500 | 1,8-2,9 | 64-87 |
| 20 persone, 100 casi diversi a testa | 2.000 | 2.000 | 1,8-2,7 | 68-80 |
| 50 persone, 40 casi diversi a testa | 2.000 | 2.000 | 2,0-2,5 | 68-81 |

- **Per l'angolo contano i casi distinti**, cioè gli errori distinti della macchina. Venti persone
  con 25 casi diversi a testa leggono l'angolo come una persona con 500 casi. Far rispondere più
  persone agli stessi casi aiuta, ma sempre meno: sugli stessi 100 casi si passa da 14 gradi di
  deviazione standard con una persona a 9 con cinque, 7 con venti, 6 con cinquanta.
- **Per il *k* del gruppo contano le persone**: cinque non bastano, per quante risposte diano,
  perché ognuna ha la sua sensibilità.
- **Se scarseggiano le chiavi**, si danno gli stessi casi a tutti. **Se scarseggiano le ore**, si
  danno casi diversi.
- **Il *k* di ciascuno** resta suo: la sensibilità della macchina viene da tutti i casi, quella
  della persona dai suoi. Con 100 casi a testa nove decimi di ciò che distingue una persona
  dall'altra è differenza vera; con 25 casi sette decimi, con i nove di ChatBench meno della metà
  [dati]. Con 25 casi il *k* di una persona ha un'incertezza di quasi un terzo: non basta per darle
  una zona.
- **L'angolo di ciascuno**, con un gruppo, non si ha: si ha quello del gruppo. Se l'angolo sia una
  proprietà del compito o della persona non lo sappiamo (paragrafo 10).

### 3.7 Il disegno a due fasi

Le ore della persona costano, quelle della macchina no. Dove esiste un archivio di casi con la chiave
(mille pratiche chiuse con il loro esito), la macchina li fa tutti. La persona ne fa 200, presi a
sorte dentro due strati: metà fra i casi che la macchina ha sbagliato, metà fra quelli che ha fatto
giusti, senza sapere quali sono quali. Ogni risposta pesa quanto i casi dell'archivio che
rappresenta. Con le stesse 200 risposte [simulazione]:

| Macchina | Archivio | Casi a sorte: angolo | Due fasi: angolo | *k*, a sorte | *k*, due fasi |
|---|---|---|---|---|---|
| all'88% | 1.000 | 55-94 | 63-86 | 1,8-2,7 | 1,7-3,0 |
| al 93% | 1.000 | 67-120 | 78-102 | 2,1-3,2 | 2,0-3,3 |
| al 98% | 1.000 (20 errori) | illeggibile | 74-109 | 2,6-4,3 | 2,6-4,1 |
| al 98% | 3.000 (60 errori) | illeggibile | 80-101 | 2,6-4,3 | 2,5-4,2 |

L'angolo guadagna quanto guadagnerebbe con due o tre volte i casi; *k* perde poco, e niente quando la
macchina è forte. Mettere nel campione tutti gli errori della macchina e pochi casi giusti non
conviene: l'angolo non migliora e *k* peggiora. Con sole 100 risposte *k* peggiora troppo (da 1,7-3,0
a 1,6-3,3). Il disegno ha un secondo vantaggio: rivedere le chiavi dei casi che la macchina ha
sbagliato è un controllo delle chiavi fatto dove serve di più.

## 4. Le regole d'uso

### 4.1 Le zone

La regola si prende dall'intervallo di *k*, per ogni tipo di caso.

| Zona | Intervallo di *k* | Tetto del margine a 90 gradi | Che cosa vuol dire |
|---|---|---|---|
| A | tutto sopra 1,5 | al più 20% | La macchina è nettamente più forte |
| B | sopra 1, non tutto sopra 1,5 | fra 20 e 41% | Più forte, non di molto |
| C | comprende 1 | 41% | Non si distinguono |
| D, E | tutto sotto 1 | come B e A, a parti scambiate | La persona è più forte |

La soglia a 1,5 è una convenzione. Corrisponde a un tetto del 20 per cento con errori indipendenti e
a un margine dell'8,5 per cento a 75 gradi, l'angolo misurato fra persone e GPT-4o. Chi vuole un tetto più
basso sposta la soglia: 1,8 per il 15 per cento, 2,2 per il 10. Nei dati dell'articolo tutte le
coppie con una macchina due volte più forte, o più, sono rimaste sotto la macchina (resa fra 0,66 e
0,76); fra 1,6 e 1,9 la resa va da 0,81 a 1,29 secondo la macchina. Due macchine sole non fissano una
soglia.

**Il margine letto in errori.** Il margine è in sensibilità, e chi lavora conta gli errori. A 90
gradi una persona al 58 per cento porta alla coppia ideale sempre lo stesso peso dell'evidenza, e il
taglio degli errori è quasi lo stesso a ogni *k*: fra il 26 e il 30 per cento degli errori della
macchina. All'angolo di 75 gradi il taglio si restringe: 17 per cento con *k* 1,2, 8 con *k* 2, 2 con
*k* 3 [simulazione]. In numeri assoluti, con una macchina al 98 per cento il tetto vale mezzo errore
ogni cento casi.

### 4.2 Che cosa i numeri permettono di dire

**In zona A.**

1. **La risposta della macchina resta, anche nel disaccordo.** Nei 464 disaccordi fra le persone e
   la chat di GPT-4o aveva ragione la chat 86 volte su cento e la persona 10. Chi l'avesse seguita
   sempre avrebbe risposto giusto 86 volte su cento; le persone si sono fermate a 77, e dove non
   l'hanno seguita avevano ragione una volta su quattro [articolo]. Nel modello, chi ha il proprio
   indizio e il solo verdetto di una macchina 2,2 volte più sensibile non può fare meglio che
   seguirla sempre [simulazione].
2. **Anche quando si è molto sicuri.** In matematica, fra chi si diceva molto sicuro, la persona
   aveva ragione in 10 disaccordi su 46 e la chat in 35 [articolo]. Il modello dà la stessa cosa: 19
   e 74 per cento. La regola «seguo la macchina, tranne quando sono molto sicuro» costa 5 punti di
   risposte giuste ai valori di GPT-4o, e 7 con una macchina al 98 per cento [simulazione].
3. **Il disaccordo è un allarme.** La chat di GPT-4o sbagliava nel 3,5 per cento degli accordi e nel
   14,4 dei disaccordi, e tre suoi errori su quattro cadevano in un disaccordo [articolo]. L'allarme
   suona nel 41 per cento dei casi, e in quei casi la macchina ha comunque ragione più di otto volte
   su dieci: serve a scegliere che cosa verificare, non a chi credere.
4. **Fare uscire la risposta.** Degli 11 punti che la coppia ha perso rispetto a GPT-4o da sola,
   quasi 8 stanno nelle chat da cui non è uscita una risposta; lì la macchina da sola rispondeva
   giusto 84 volte su cento e le persone sono rimaste al 50 [articolo]. È una descrizione, non una
   causa provata, ma il rimedio non costa: dare alla macchina il caso intero, con le opzioni, e se
   non risponde richiedere, senza ripiegare sulla propria risposta.
5. **Controllare la macchina non aggiunge sensibilità.** Con una resa di 0,76 la coppia dà poco più
   della metà del peso dell'evidenza della macchina da sola, anche a parità di tempo [articolo]. Se
   la firma resta, è per responsabilità, o perché i due errori non costano uguale (paragrafo 7).

**In zona C.**

1. **Nessuno dei due comanda.** Sulle domande di etica i disaccordi si dividono quasi a metà: 49 e
   37 per cento con GPT-4o, 43 e 52 con Llama [articolo].
2. **L'accordo è la parte utile.** Nel modello, alla pari, la macchina sbaglia nel 9 per cento degli
   accordi e nel 52 dei disaccordi [simulazione].
3. **Un disaccordo si decide con altro**: più tempo, una terza fonte.

**In zona B** i numeri non danno una regola. Il modello promette alla coppia ideale fra l'8 e il 40
per cento di sensibilità, secondo *k* e l'angolo, e i dati dell'articolo a questi valori di *k* vanno
in tutti e due i versi.
Si aggiungono casi, e si decide sulla prova della coppia.

**Nelle zone D ed E**, con la persona più forte, c'è un dato solo e va nel verso della prudenza: con
Llama sull'etica (*k* 0,49) le persone sono scese a 0,91 turing, e hanno seguito le risposte
sbagliate della macchina il 62 per cento delle volte contro il 26 del modello [articolo].

**Che cosa rende ogni regola, nel modello.** Risposte giuste su cento, quattro opzioni [simulazione,
`sim_rules.py`]. La sicurezza della persona è qui la distanza fra il suo primo e il suo secondo
indizio: una persona che sa esattamente quanto è sicura.

| Situazione | Persona | Macchina | Coppia ideale | Pesi uguali | Pesa al meglio il verdetto | «Macchina, tranne se molto sicuro» |
|---|---|---|---|---|---|---|
| GPT-4o su ChatBench (*k* 2,16) | 57,7 | 88,3 | 89,0 | 85,1 | 88,3 | 83,4 |
| Macchina al 93% (*k* 2,5, 90 gradi) | 57,7 | 93,0 | 94,9 | 92,7 | 93,1 | 87,4 |
| Macchina al 98% (*k* 3,2, 90 gradi) | 57,7 | 98,0 | 98,5 | 96,9 | 98,0 | 90,7 |
| Zona di mezzo (*k* 1,5) | 64,5 | 81,2 | 84,5 | 83,2 | 82,2 | 81,0 |
| Llama 3.1 8B (*k* 1,18) | 60,5 | 66,7 | 73,2 | 73,0 | 70,5 | 69,2 |
| Alla pari (*k* 1,08) | 71,2 | 74,5 | 81,2 | 81,2 | 79,1 | 77,5 |

Due letture. Con una macchina forte il margine si prende solo mettendo in comune gli indizi
graduati: con il solo verdetto il meglio è seguirla, e ogni regola che lascia decidere la persona
perde punti. Alla pari anche una regola rozza guadagna 3 punti, e l'ideale quasi 7.

### 4.3 Che cosa i numeri non permettono di dire

- **«Sotto l'angolo utile si segue la macchina.»** Vero nel modello, inservibile come regola: con 100
  casi non si sa da che parte dell'angolo utile si sta (lo si decide una volta su sette ai valori di
  GPT-4o, meno di una su due con 400 casi). E con GPT-4o la coppia vera ha perso un quarto della
  sensibilità della macchina stando *sopra* l'angolo utile, 75 gradi contro 62. A decidere è *k*.
- **I pesi della coppia ideale.** Il rapporto vale 4,5 all'angolo misurato con GPT-4o e 2,5 a 86
  gradi, l'estremo alto dell'intervallo, mentre l'estremo basso sta già sotto l'angolo utile
  [articolo]. Un numero che cambia così in un intervallo ottenuto con 1.500 prove non è una regola
  per chi ne ha 100.
- **«Alla pari si pesano i pareri con la sicurezza.»** È ciò che dice il modello, non ciò che si è
  visto. Sull'etica la resa è 1,00 con un intervallo da 0,66 a 1,47; tre disaccordi su quattro
  vengono da persone molto sicure, e nei disaccordi sull'etica la persona ha ragione 37 volte su
  cento [articolo]. Nel modello, alla pari, chi è molto sicuro ha ragione 70 volte su cento
  [simulazione]: la sicurezza dichiarata non è quella del modello. Va provato, non prescritto.
- **«Nel disaccordo, ricontrolla.»** Ricontrollare da soli, nei dati, ha fatto peggio che seguire.
  Che cosa renda una terza fonte non lo dice nessun dato dell'articolo.
- **Una regola dalla propria sicurezza.** Con 100 casi i disaccordi sono una quarantina, divisi su
  tre livelli: troppo pochi per fissare una soglia personale.
- **Un guadagno oltre il modello.** Che le persone sappiano quando la macchina sbaglia non è
  provato: seguono la chat più spesso quando ha ragione, ma lo farebbe anche chi la pesa sempre allo
  stesso modo [articolo].
- **Il tempo.** Senza cronometro la formula della potenza resta vuota.
- **Un turing senza *k*.** Chi ricopia una macchina *k* volte più sensibile vale *k* turing
  [articolo].

## 5. La prova della coppia

**Sugli stessi casi della taratura** (terzo passo della sessione) si leggono il turing, la resa, e
dove vanno i punti fra la macchina da sola e la coppia: la conversazione, la decisione, le chat senza
risposta. Quanti casi servono [simulazione, una persona]:

| Coppia vera | 100 casi | 200 casi | 400 casi |
|---|---|---|---|
| Come con GPT-4o: turing 1,64, resa 0,76 | resa 0,60-0,93; «sotto la macchina» deciso nel 68% | 0,65-0,88; 95% | 0,68-0,84; 100% |
| Alla pari, coppia ideale: turing 1,27, resa 1,17 | resa 0,98-1,41; «sopra la macchina» nel 39% | 1,04-1,33; 72% | 1,07-1,28; 93% |
| Alla pari, la persona ascolta poco: resa 1,07 | 0,85-1,37; 8% | 0,91-1,26; 11% | 0,95-1,21; 21% |

Che una coppia stia sotto la macchina si vede con 200 casi quando la distanza è quella di ChatBench.
Che una coppia alla pari batta tutti e due chiede 400 casi se la coppia è ideale; se guadagna meno
non si vede. Il turing sopra uno si vede prima: con 100 casi sempre nel primo caso, nel 64 per cento
nel secondo.

**La prova della regola**, sul modulo B. Fissata la regola, la persona rifà il giro (da sola, poi
con la macchina applicando la regola) su casi nuovi. Le risposte date da sola si aggiungono alla
taratura: dopo il modulo B i casi per *k* e per l'angolo sono 200. La resa del modulo B si confronta
con quella del modulo A, e la differenza fra due rese è più incerta di ciascuna: con 100 casi per
modulo si vede solo un salto di 0,33 (da 0,76 a più di 1,05), con 200 di 0,23, con 400 di 0,17
[simulazione]. Un miglioramento piccolo, una persona sola non lo vede.

**Il tempo.** Con i due tempi per caso, la potenza della coppia rispetto alla persona è il turing al
quadrato per il rapporto dei tempi; rispetto alla macchina a cui si delega senza guardare, la resa
al quadrato per il rapporto dei tempi [articolo, formule 8 e 9]. Controllare si ripaga solo se la
resa supera la radice del rapporto fra il tempo del controllo e quello della delega: con un controllo
che dura sei volte la delega, una resa di 2,4.

## 6. La ripetizione

**La macchina.** A ogni versione nuova si ridanno alla macchina i moduli già usati: le risposte
della persona sono in archivio, e *k*, angolo e allarme si rifanno senza un'ora di lavoro. Vale se i
casi non sono finiti nell'addestramento della versione nuova: dipende dalle condizioni con cui sono
stati inviati. Fra una versione e l'altra basta una sentinella: trenta casi ridati ogni mese. Con il
rumore fra due chiamate di GPT-4o su ChatBench (una risposta su venti cambia esito) sei o più
risposte cambiate su trenta capitano meno di una volta su cento [dati]: sono il segnale per rifare
la taratura.

**La persona.** Si misura su casi nuovi (il modulo C), con il turing ancorato al primo giorno e il
rapporto fra la sensibilità da sola di allora e di oggi [articolo]. Ma una persona sola vede poco
[simulazione, persona al 58 per cento, due moduli diversi]:

| Calo della sensibilità da sola | Risposte giuste, da 58% a | 100 risposte per volta | 200 | 400 | 1.000 | 3.000 |
|---|---|---|---|---|---|---|
| 10% | 54,3% | 8% | 10% | 16% | 34% | 76% |
| 20% | 50,9% | 17% | 28% | 50% | 87% | 100% |
| 30% | 47,4% | 33% | 54% | 84% | 100% | 100% |
| 40% | 44,0% | 52% | 79% | 98% | 100% | 100% |

Le percentuali sono la probabilità di accorgersi del calo. Con 100 casi per volta una persona vede,
una volta su due, solo un calo del 40 per cento. Un'organizzazione che mette insieme 1.000 risposte
per volta (dieci persone con 100 casi) vede un calo del 20 per cento quasi nove volte su dieci, e con
3.000 uno del 10 tre volte su quattro, se il calo è di tutti. Il turing ancorato è una misura da
organizzazione.

**Quando.** La macchina: a ogni versione, e quando la sentinella suona. La persona: una volta
l'anno, o quando cambia il modo di lavorare. I moduli si consumano: tre moduli sono tre misure.

## 7. Le insidie

| Insidia | Quanto sposta | Che cosa fare |
|---|---|---|
| **Chiavi sbagliate** | Macchina all'88%: con il 2% di chiavi sbagliate l'angolo passa da 74 a 71 gradi, con il 5% a 66. Macchina al 98%: con l'1% da 90 a 80 gradi, con il 2% a 74, e *k* da 3,2 a 2,9; un terzo e poi metà dei suoi «errori» sono errori della chiave [simulazione] | Seconda lettura cieca su un decimo delle chiavi e su tutti i casi che la macchina sbaglia; riportare la quota che non regge |
| **Casi già visti dal modello** | Macchina vera all'80%: se quattro casi su dieci li conosce, si misura l'88% e *k* passa da 1,76 a 2,15. Alla pari, con sei casi su dieci visti, *k* passa da 1,08 a 1,59: da pari a macchina più forte [simulazione] | Casi propri, mai pubblicati, successivi all'addestramento; il modulo C non va inviato prima dell'uso |
| **Tipi di caso mescolati** | GPT-4o vale 3,18 volte le persone in fisica e 1,08 in etica [articolo]; nel foglio d'esempio il totale dà zona A, un tipo di caso dà zona C | Un *k* e una regola per tipo; almeno 100 casi per tipo |
| **Casi troppo difficili per la persona** | Con 100 casi, l'errore tipico della sensibilità è il 15% al 58 per cento di risposte giuste, il 30% al 40, il 42% al 35 [simulazione]: *k* e turing esplodono | Scegliere tipi di caso in cui la persona sta almeno al 50% su quattro opzioni (75% sul sì/no) |
| **Casi troppo facili** | Al 98% una taratura da 100 casi su otto non vede errori; l'angolo non esiste | Minimo di *k* e tetto del margine; due fasi |
| **Versione, richiesta, temperatura** | Su ChatBench, fra due chiamate uguali, GPT-4o cambia esito su una domanda su venti e Llama su una su cinque; ma otto decimi dell'errore di GPT-4o si ripetono, e la maggioranza di cinque chiamate guadagna mezzo punto [dati] | Annotare versione datata, richiesta, impostazioni. Chiedere due volte dice dove la macchina è instabile, non corregge i suoi errori |
| **La persona che ricorda i casi** | Se ricorda la chiave di tre casi su dieci, *k* passa da 2,16 a 1,55 [simulazione] | Ogni modulo una volta sola; nessuna chiave prima della fine |
| **La macchina in chat** | Llama: 85% in chat, 73% da sola. GPT-4o: più di un quinto delle prove senza risposta [articolo] | La prova della coppia, con la risposta della chat annotata caso per caso |
| **I due errori non costano uguale** | La sensibilità non lo vede. Esempio di conto: positivi uno su dieci, un'omissione costa venti falsi allarmi. Una coppia con resa 0,76 che sposta il criterio costa 48 ogni cento casi; la macchina con un criterio neutro 35, con un criterio prudente che non si può spostare 55 [simulazione] | Sul sì/no contare a parte omissioni e falsi allarmi di persona, macchina e coppia, e pesarli con i propri costi e con la frequenza vera dei positivi |
| **Casi legati fra loro** | Dieci pratiche dello stesso cliente non sono dieci casi: gli intervalli veri sono più larghi | Un caso per cliente o per pratica; altrimenti ricampionare per gruppi |
| **Guardare e aggiungere** | Allungare la taratura finché esce la zona sperata alza la probabilità di sbagliare zona | Fissare prima i passi (100, poi 200) e la zona che fa fermare |

Sulle chiavi c'è un riscontro nei dati. Su αNLI, dove in una storia su dieci la maggioranza dei
votanti contraddice la chiave, l'angolo fra le persone e Fable 5.2 vale 47 gradi con tutte le storie
e 68 senza quelle contestate [articolo]. In simulazione, su casi a due opzioni con una persona al 90
per cento e una macchina al 96, il 10 per cento di chiavi sbagliate porta l'angolo da 68 a 33 gradi,
e il 2 per cento a 53.

Una chiave contestata non è per forza sbagliata. Dove il secondo lettore non conferma la chiave, i
conti si danno due volte, con tutti i casi e senza i contestati, come fa l'articolo. Le due letture
sono due estremi: togliere casi guardando le risposte di chi si sta misurando apre l'angolo anche
da solo [articolo].

## 8. Chi fa che cosa

**Chi produce il modello** non può pubblicare *k*, che non è suo. Può pubblicare ciò che permette a
chiunque di calcolarlo su un test aperto:

- **le risposte domanda per domanda**, non solo l'accuratezza: identificativo della domanda,
  versione del test e della chiave, risposta data, e la distribuzione delle risposte su più chiamate
  (ChatBench ne ha cinquanta per domanda);
- **le condizioni**: versione datata, testo della richiesta, temperatura, strumenti accesi, data;
- **su test che hanno le risposte delle persone caso per caso**, perché l'angolo con una
  popolazione di riferimento si calcola solo così; e l'angolo con la versione precedente, che dice
  quanto invecchia una taratura (Opus 5.5 e Fable 5.2 hanno una correlazione di 0,95 [articolo]);
- **quanto spesso non esce una risposta utilizzabile** in un uso da chat: nei dati è la perdita più
  grande, e nessuna scheda di modello la riporta;
- **risultati su domande nuove**, tenute da un terzo e pubblicate dopo il ritiro: un test aperto e
  ripetibile è anche un test che il modello successivo ha letto.

Con queste risposte pubbliche, a chi usa il modello resta da fare solo la propria metà: rispondere a
100 o 200 domande di quel test, e avere *k* e l'allarme senza interrogare la macchina. È una stima
per eccesso, perché il test è pubblico.

**Chi usa il modello** misura ciò che è suo: la propria sensibilità sui propri casi, la sicurezza,
il tempo, la coppia. Tiene il foglio e lo rifà a ogni versione.

**Un'organizzazione** fa ciò che una persona sola non può:

- l'archivio dei casi con la chiave, la seconda lettura, i moduli tirati a sorte, il segreto;
- la macchina interrogata una volta per tutti, con le condizioni fissate e le risposte conservate;
- il disegno a due fasi e il gruppo, che danno l'angolo;
- un *k* per tipo di caso e per fascia di esperienza, e la regola che ne segue: chi firma, quando
  una verifica è obbligatoria, a chi va un disaccordo;
- la ripetizione, la sentinella, il turing ancorato sul gruppo;
- la decisione, presa prima, su chi vede i risultati di ciascuno: sono misure di una persona, e con
  poche decine di casi a testa sono troppo incerte per giudicarla.

## 9. La forma

**La scheda.** Una pagina: [`PAIR-CARD.it.md`](PAIR-CARD.it.md). Che cosa serve, i quattro passi, che
cosa si legge con 100 e con 200 casi, la regola per zona, che cosa non leggere, quando rifarla,
cinque errori.

**Il foglio.** Un CSV con una riga per caso: [`examples/pair-sheet-template.csv`](examples/pair-sheet-template.csv).
Colonne: `case`, `stratum`, `key`, `person`, `confidence`, `machine`, `chat`, `pair`, `seconds_alone`,
`seconds_pair`, e `weight` per il disegno a due fasi. Obbligatorie solo `key`, `person`, `machine`.
Il programma accetta anche i nomi italiani delle colonne (`caso`, `strato`, `chiave`, `persona`,
`sicurezza`, `macchina`, `chat`, `coppia`, `secondi_solo`, `secondi_coppia`, `peso`).

**Il programma.** `turing-meter pair`, nel pacchetto del repository:

    turing-meter pair docs/examples/pair-sheet-example.csv --options 4 --delegate-seconds 20

Restituisce, in inglese, le due sensibilità, *k* con l'intervallo (ricampionamento dei casi, ed
esatto quando gli errori sono pochi), l'angolo con l'intervallo e l'avviso «indicativo» o «non
stimabile», l'angolo utile, il margine, il tetto, gli stessi numeri in errori su cento, l'allarme, la
tabella per sicurezza, turing e resa, dove vanno i punti, la potenza, *k* per tipo di caso con
l'avviso se le zone divergono, la zona e la regola. Un referto completo è in
[`examples/pair-report-example.txt`](examples/pair-report-example.txt) (foglio finto, 200 casi,
[`examples/pair-sheet-example.csv`](examples/pair-sheet-example.csv)). L'avviso sui tipi di caso lì
scatta: sul totale zona A, sui casi atipici (`atypical`) zona C. Con `--json file` scrive anche la
misura in JSON.

Alla prova su fogli di cui si conosce il vero (`analysis/pair-calibration/check_pair_tool.py`, e in
piccolo i test di `tests/test_pair.py`): con 6.000 casi ritrova *k* 2,16, angolo 74,4 e margine 2,0
per cento; sul sì/no con un'inclinazione comune non ha errore sistematico; con i pesi delle due fasi
restringe l'intervallo dell'angolo da 39 a 23 gradi; con zero errori della macchina dà un minimo per
*k* e non riporta l'angolo.

**Che cosa c'era già e che cosa si è aggiunto.** I conti dell'articolo stanno in
`experiments/angle-between-errors/angle/` e quella cartella resta com'è. Il pacchetto
`src/turing_meter` ha preso da lì le funzioni a *m* opzioni (`mchoice.py`, copia di `angle/sdt.py`) e
ci ha costruito sopra `pair.py`.

| Serve | Nell'esperimento | Aggiunto nel pacchetto |
|---|---|---|
| Sensibilità a *m* opzioni, tetracorica, ideale, angolo utile | `angle/sdt.py` | Portate in `mchoice.py`, invariate |
| Conversione dalla correlazione delle risposte giuste a quella degli indizi | `sdt.convert_rho_mafc`, `measure.conversion_table` | Per una persona sola si usa la prima, con la soglia esatta. La seconda arrotonda le soglie, un'approssimazione: con una macchina intorno al 98% la sua tabella smette di crescere con la correlazione e non si inverte più in modo affidabile |
| Angolo e intervalli per molte persone con pochi casi | `measure.compare` | Per una persona sola `measure.compare` scarta un ricampionamento su tre e non ha intervalli esatti per pochi errori: qui ricampionamento dei casi e intervalli esatti |
| Minimo di *k* con zero errori | `conversations.zero_errors` | Una funzione generale |
| Margine con intervallo, tetto, errori su cento | Solo la formula | Tutto |
| Accordo, disaccordo, sicurezza, dove vanno i punti | `conversations.py`, legato alle colonne di ChatBench | La versione su un foglio qualunque |
| Pesi per le due fasi | `_one_pass` accetta pesi | Il ricampionamento per strato con i pesi |
| Lettura del foglio, referto, zone | Niente | Tutto |
| Sì/no per una persona, con i due tipi di errore | `measure._one_pass` con le classi | Le sensibilità e la correlazione per classe |
| Gruppo: *k* di ciascuno riportato verso la media | `sdt.shrink` per le soglie | Solo in simulazione (`sim_group.py`) |
| Prove | `tests/` dell'esperimento | `tests/test_pair.py`, e la copertura dell'intervallo per una persona sola in `analysis/pair-calibration/` |

**Dove sta.** Non nella cartella dell'esperimento: quella è il verbale dell'articolo, con il
protocollo scritto prima, le deviazioni datate, i risultati citati. Uno strumento per chi usa il
modello porterebbe stimatori che cambiano (la soglia esatta, gli intervalli esatti) accanto a quelli
che hanno prodotto i numeri pubblicati. Il programma è nel pacchetto (`src/turing_meter/pair.py`,
comando `turing-meter pair`), l'esperimento tiene la sua copia congelata di `sdt.py`, le simulazioni
di queste pagine stanno in `analysis/pair-calibration/`, il protocollo e la scheda in `docs/`.

## 10. Che cosa non sappiamo

1. **Se l'angolo appartiene al compito o alla persona.** Nei dati nessuno ha abbastanza casi per un
   angolo suo. Usare l'angolo del gruppo per una persona è un'ipotesi.
2. **Se una coppia vera prende il margine quando c'è.** Alla pari il modello promette fra 3 e 7
   punti; nei dati la resa non si distingue da uno. Se pesare i pareri con la sicurezza dichiarata
   funzioni con persone vere, che non sono tarate come il modello suppone, nessuno l'ha misurato.
3. **Che cosa fare quando l'allarme suona.** I dati dicono che cosa non funziona, decidere da soli.
   Che cosa renda una terza fonte, o un secondo modello, non c'è.
4. **Se serve la sicurezza della macchina.** Il margine in zona A si prende solo con un indizio
   graduato della macchina. Se la sicurezza che un modello dichiara lo sia, qui non è misurato.
5. **I modelli di oggi.** Nessuna persona ha lavorato con loro nei dati: per loro c'è *k*, non un
   turing, e sulle domande d'esame l'angolo non si stima. Le zone, per i modelli che si usano
   davvero, vengono da *k* solo.
6. **La chat di oggi.** Memoria, strumenti, ricerca in rete: se le chat di oggi restino mute quanto
   quelle di ChatBench, o assecondino di più, non lo sappiamo. Né sappiamo se il terzo in più di
   Llama in chat sia merito di chi chiede.
7. **Le persone.** I dati sono di lavoratori reclutati in rete su compiti brevi. Per uno
   specialista sul suo lavoro non c'è una sola misura.
8. **Il tempo e i costi.** Lo strumento legge la sensibilità per decisione. Nessun dato
   dell'articolo ha i tempi caso per caso, né i costi dei due errori.
9. **Il lavoro senza chiave.** Risposte aperte, compiti lunghi, scelte di che cosa chiedere: fuori
   dallo strumento, e le convenzioni per farcelo entrare cambiano *k* da 1,3 a 5,8.
10. **Quanto dura una taratura.** Quanto cambiano *k* e l'angolo da una versione all'altra, e se
    l'angolo di una persona si chiude con l'uso: lo studio nel tempo non è stato fatto.
11. **La soglia delle zone.** 1,5 è una convenzione, appoggiata a due macchine.
12. **Quanto valgono le simulazioni fuori da ChatBench.** I conti suppongono indizi gaussiani e
    casi indipendenti. Sui dati di ChatBench le oscillazioni di *k* e dell'angolo tornano; su altri
    compiti non è stato provato, e l'intervallo per una persona sola è verificato solo in
    simulazione.
13. **Quante chiavi sono sbagliate**, in un archivio vero: si sa solo dopo la seconda lettura.

## Appendice. Come rifare i conti

Gli script stanno in [`analysis/pair-calibration/`](../analysis/pair-calibration/) e leggono gli
stimatori dal repository dell'esperimento senza scrivere niente al suo interno; il percorso si cambia
con la variabile `TURING_METER_EXPERIMENT`. Servono Python 3.11 o più, numpy e scipy (con il
repository: `uv run python analysis/pair-calibration/sim_sample.py`), più pandas e i dati di ChatBench
(`python -m angle.download data chatbench`, dentro l'esperimento) per `verify_chatbench.py`. Tutto
insieme gira in una ventina di minuti su quattro processori; la tabella di conversione
(`conversion-m4.npz`) si calcola la prima volta, in sei minuti. Il README della cartella dice quale
comando scrive quale file.

| Script | Che cosa fa | Risultati (`analysis/pair-calibration/results/`) |
|---|---|---|
| `estimators.py` | Gli stimatori del repository in forma vettoriale, la tabella di conversione su una griglia | |
| `sim_sample.py` | Una persona sola, da 50 a 1.600 casi, sei situazioni: stime, intervalli, copertura, letture | `sample.txt` |
| `sim_sample.py --errors` | La precisione dell'angolo per numero di errori della macchina | `machine-errors.txt` |
| `sim_sample.py --verify` | La via rapida contro `measure.compare`: stesso *k*, angolo entro un grado in mediana, stessa larghezza dell'intervallo | `verification.txt` |
| `sim_rules.py` | L'allarme e le regole nel modello | `rules.txt` |
| `sim_pair.py` | Turing e resa di una persona; la ripetizione | `pair.txt` |
| `sim_group.py` | Una persona contro un gruppo, con `measure._one_pass` | `group.txt` |
| `sim_two_phase.py` | Il disegno a due fasi | `two-phase.txt` |
| `sim_pitfalls.py` | Chiavi, contaminazione, casi difficili, sì/no, risposte aperte | `pitfalls.txt` |
| `verify_chatbench.py` | Le tarature estratte dai dati veri; la macchina che cambia risposta | `chatbench-check.txt` |
| `minor_figures.py` | Costi dei due errori, sentinella, memoria, affidabilità, margine in errori | `minor-figures.txt` |
| `check_pair_tool.py` | `turing-meter pair` alla prova su fogli di cui si conosce il vero | `tool-check.txt` |
| `make_example.py` | Il foglio finto di `docs/examples/` | `docs/examples/pair-sheet-example.csv` |

Due avvertenze sui conti. La via rapida usa la tetracorica con la soglia esatta e la tabella di
`sdt.convert_rho_mafc`; lo strumento dell'articolo, pensato per molte persone con pochi casi, usa
soglie personali arrotondate. Sulle stesse tarature simulate i due danno lo stesso *k* e angoli che
differiscono di un grado o meno in mediana, con un massimo di dieci in un caso su 150, dove la
macchina era al 93 per cento. E tutte le percentuali «deciso» dipendono dalla soglia e dal livello
dell'intervallo: sono ordini di grandezza, non garanzie.
