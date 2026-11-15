# Prima di fidarsi di un modello: la taratura in una pagina

> [English](PAIR-CARD.md) · **Italiano**

Bozza del 15 novembre 2026. Il dettaglio, con i conti, è in [`PAIR-PROTOCOL.it.md`](PAIR-PROTOCOL.it.md).

**A che cosa serve.** A sapere, per un tipo di lavoro, quante volte la macchina vale te (*k*), e a ricavarne una regola d'uso. Vale per decisioni con una risposta giusta verificabile. Non dice niente del lavoro senza una chiave.

## Che cosa serve

- **300 casi tuoi per tipo di lavoro**, divisi a sorte in tre moduli da 100: uno per la taratura, uno per provare la regola, uno da parte per rifarla fra un anno. Presi a sorte dal lavoro vero, mai pubblicati, con la risposta giusta verificata da qualcuno che non sei tu e non è la macchina. In forma chiusa, a quattro opzioni, se si può. Cento casi sono il minimo per cominciare.
- **Un modello fissato**: la versione con la data, il testo della richiesta, le impostazioni. Tutto scritto.
- **Un foglio** con una riga per caso ([`examples/pair-sheet-template.csv`](examples/pair-sheet-template.csv)) e il programma `turing-meter pair foglio.csv --options 4` (il referto è in inglese).

## Che cosa fai

1. **Da solo.** Rispondi a tutti i casi del modulo senza vedere né la chiave né la macchina. Per ognuno segni quanto sei sicuro (1, 2, 3) e il tempo.
2. **La macchina da sola.** Gli stessi casi, uno per chiamata, sempre con la stessa richiesta. Meglio tre chiamate per caso: si vede dove cambia risposta.
3. **Insieme.** Rifai i casi con la chat, come la useresti davvero. Segni la risposta uscita dalla chat (o che non è uscita) e la tua risposta finale.
4. **Solo alla fine** guardi chiavi e risultati. Quei casi, per te, sono spesi: non si riusano.

## Che cosa leggi

| Numero | Che cosa dice | Con 100 casi | Con 200 (dopo il secondo modulo) |
|---|---|---|---|
| *k*, con il suo intervallo | Quante volte la macchina vale te | Da 1,6 a 3,1 se vale 2,2 | Da 1,75 a 2,8 |
| Tetto del margine | Il massimo che una coppia perfetta aggiungerebbe al più forte | Sì | Sì |
| Allarme | Quanto sbaglia la macchina dove siete d'accordo e dove no | Sì, a grandi linee | Sì |
| Resa sulla macchina | Se insieme fate meglio o peggio della macchina da sola | Solo se la distanza è grande | Sì |
| Angolo fra gli errori | Quanto sbagliate sugli stessi casi | **No** (da 47 a 105 gradi) | **No** (da 56 a 95) |

L'angolo vive sugli errori della macchina: per conoscerlo entro dieci gradi ne servono fra 50 e 80. Sotto i 40 errori è solo un'indicazione, sotto i 10 non va letto.

## La regola, per zona

- **Zona A: tutto l'intervallo di *k* sopra 1,5.** La risposta della macchina resta, anche quando non sei d'accordo e anche quando sei molto sicuro. Il disaccordo è un allarme: quei casi vanno a una terza fonte, non alla tua seconda opinione. Fai uscire la risposta: caso intero nella chat, e se non risponde richiedi.
- **Zona B: *k* sopra 1, non tutto sopra 1,5.** Nessuna regola fissa. Aggiungi casi e decidi sulla prova insieme (punto 3).
- **Zona C: l'intervallo comprende 1.** Nessuno dei due comanda. In accordo si procede; un disaccordo si decide con altro (tempo, una terza fonte).
- **Zone D ed E: sei tu il più forte.** Le parti si scambiano. Attenzione a non farti trascinare da una risposta sbagliata detta bene.

Se i tipi di caso danno zone diverse, la regola si dà tipo per tipo.

## Che cosa non leggere

- Un *k* complessivo, se i tipi di caso divergono.
- Un turing senza *k* accanto: chi ricopia una macchina *k* volte più forte vale *k* turing.
- La resa della taratura come promessa per la chat: nella chat la macchina non è la stessa.

## Quando rifarla

- **A ogni versione nuova del modello**: solo la macchina, sugli stessi casi (se non sono finiti nel suo addestramento).
- **Ogni 6-12 mesi, o quando cambia il tuo modo di lavorare**: tu, su casi nuovi tenuti da parte.

## Cinque errori che rovinano la misura

1. Casi presi da test pubblici: la macchina può averli già visti, e *k* sale da solo.
2. Chiavi non verificate: con una macchina al 98 per cento, una chiave sbagliata su cento sposta l'angolo di dieci gradi.
3. Casi scelti a mano, perché interessanti: misurano i casi interessanti.
4. Guardare la risposta della macchina prima di dare la tua.
5. Riusare i casi: se ne ricordi tre su dieci, *k* passa da 2,2 a 1,6.
