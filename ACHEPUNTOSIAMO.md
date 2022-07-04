## 4/07

-abbiamo capito che fanno le classi GN e OGN in models.py
- nella demo siamo arrivati a studiare la loss ma non capiamo bene la regolarizzazione
(perchè si def una nuova loss e non lo fa in models? perchè normalizza in quel modo?)
- bisogna molto sfoltire il codice: la regolarizzazione KL non funziona bene quindi va buttata giù insieme a un miliardo di altre cose.
- vogliamo conservare la regolarizzazione L1, bottleneck e si potrebbe aggiungere L2.
- attenzione nel numero di batch ed nel plot dei risultati se cambi il numero di nodi potrebbe sfanculare.
- aggiunto tutorial pysr
