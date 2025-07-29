# GasRL
This is the code to easy replicate the results of [inserisci link]
Di seguito i passaggi necessari alla comprensione ed all'utilizzo del codice:

1) Easy_Train.py permette l'allenamento del modello per il numero di step desidarti, salvando il modello e testandolo a diversi checpoint intermedi della trining-phase. Modificando i valori inseriti nel "main" al termine dello script è possibile modificare i diversi parametri, come il numero complessivo di step di allenamento e l'intervallo di salvataggio

2) Easy_Plot.py contiene semplicemente le funzioni necessarie ad utilizzare correttamente gli altri file presenti

3) Easy_Test.jpynb permette di ottenere e salvare i dati aggregati e relativi alle singole timeseries per le diverse metriche e deve essere runnato prima di entrambi gli altri due notebook

4) Easy_Price.jpynb calcola la volatilità di prezzo con i due approcci proposti ed offre un confronto grafico tra i due

5) Easy_Seasonality.jpynb permette di calcolare la stagionalità dei prezzi secondo i diversi approcci proposti nell'articolo e di confrontarla con quella ottenuta dai dati reali
