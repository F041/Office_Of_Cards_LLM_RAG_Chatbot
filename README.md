# Office Of Cards LLM RAG ChatBot

![Office Of Cards Logo](Logo.png)

## Descrizione
Office Of Cards LLM RAG ChatBot è un'applicazione web interattiva che fornisce un assistente virtuale basato su intelligenza artificiale per rispondere alle domande degli utenti riguardo al contenuto del [canale YouTube](https://www.youtube.com/@OfficeofCards), [podcast](https://open.spotify.com/show/2cqzDBQRxqgba39VPp3FDs) e [libro](https://amzn.to/3VG7ifT) "Office Of Cards" di Davide Cervellin. L'applicazione utilizza Streamlit per l'interfaccia utente e GPT-4 per la generazione delle risposte.
Il vectorDB viene popolato dalle trascrizione YouTube tramite langchain e poi

## Requisiti
- Python 3.x
- Streamlit
- OpenAI API Key
- Pacchetti Python aggiuntivi specificati nel file `requirements.txt`

## Installazione
1. Clona il repository sul tuo computer.
2. Crea un nuovo ambiente virtuale ed installa le dipendenze tramite il comando `pip install -r requirements.txt`.
3. Imposta la chiave API di OpenAI nel file `.stramlit/secrets.toml`.
4. Avvia l'applicazione eseguendo il comando `streamlit run app.py`.

## Utilizzo
1. Accedi all'applicazione tramite il link locale fornito da Streamlit.
2. Interagisci con l'assistente virtuale scrivendo domande nella chat.
3. Ricevi risposte generate dall'intelligenza artificiale, accompagnate da fonti pertinenti.

## Contributi
Siamo aperti ai contributi! Se desideri contribuire a questo progetto contatta il team di sviluppo:
- [Simone Cecconi](mailto:smn.ccc@gmail.com)

## Licenza
Questo progetto è concesso in licenza sotto i termini della licenza [MIT](LICENSE.md).
