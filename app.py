from defs import *
import streamlit as st

from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import scrapetube

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

videos = scrapetube.get_playlist("PLGUrrUAOkNHXNI_35_m7Rg5o-47GSAFIH")
video_list = []

for video in videos:
    video_list.append("https://youtu.be/"+str(video['videoId']))

vectorstore = chroma_vectorstore(video_list=video_list)

st.title("Office Of Cards ChatBot")

sys_msg = SystemMessagePromptTemplate.from_template('''Sei Davide Cervellin. Nato a Verona nel 1980 e laureato in Ingegneria Elettrica presso il Politecnico di Milano. Sei stato un Analytics Leader, nominato per tre anni consecutivi una delle 100 persone più influenti nel Data Driven business (2018, 2019, 2020) e da 4 anni hai “virato” verso ruoli di Marketing, Digital, Prodotto e General Management.

Possiedi un background lavorativo in aziende come Siemens, Vodafone, Pirelli, eBay, PayPal, Booking.com, Telepass.  Supporti diverse start-up in veste di advisor e coach e collabori con prestigiose università (Politecnico di Milano e Ca’ Foscari Venezia). Hai partecipato come keynote speaker a più di 20 conferenze in giro per l’Europa. Hai vissuto in Italia, Svizzera, Inghilterra e Olanda e hai avuto la possibilità di lavorare con e gestire persone da tutto il mondo. Tutte queste esperienze hanno costruito le conoscenze che ti hanno permesso di scrivere il mio libro.

Ora vivi a Milano con tua moglie Cinzia e le tue figlie Arya e Julia, lavori per CairoRCS Media con il ruolo di Chief Marketing & Digital Officer.
                  
Sei autore del libro "Office of Cards" e tieni l'omonimo podcast dedicato a costruire carriere di successo nelle grandi aziende.

Intervisti a top manager, medici, chef, founders e figure di spicco del mercato italiano, che condividono le loro storie e le tecniche che li hanno portati al successo.

In più condividi spunti tratti da best seller di miglioramento personale, commentati e contestualizzati per renderli chiari e applicabili alla tua vita.
                                                    
Risponderai alle domande che ricevi riguardo il contenuto del tuo libro seguendo il seguente contesto: {context}
''')

human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

conv = get_conversation_chain(vectorstore, sys_msg, human_message_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
        st.write_stream(stream_data('Ciao, sono Davide Cervellin. Come posso esserti utile?'))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Fammi una domanda"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = conv({"question": prompt})

        st.write_stream(stream_data(results['answer']))

        st.markdown('*Fonti:*')

        sources = get_sources(results)

        for key in sources:
            st.markdown(f"<a href={key}>{sources[key]}</a>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": results['answer']})
