Questa repository contiene una semplice applicazione interattiva in cui dimostra come, effettuando una corretta calibrazione della fotocamera in uso (tramite l'utilizzo di una "schacchiera" con celle quadrate bianco/nere) è possibile ricostruire la terza dimesione (profondità) persa a causa della proiezione dell'immagine dal mondo reale tridimensionale al piano d'immagine bidimensionale.

L'applicazione permette di disegnare dei solidi sovraimposti sulla schacchiera, dando la possibilità di scegliere alcune caratteristiche del solido stesso (base, altezza, raggio...) L'unità di misura è la base di una cella.

## Prerequisiti

streamlit, numpy, matplotlib

## Esecuzione del programma

streamlit run app.py

### Notebook interattivo

Nota: nella cartella Scripts sono presenti dei notebook in inglese che dettagliano la fase della calibazione in sé.
