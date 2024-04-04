# Import necessary libraries and modules
from defs import *
import streamlit as st

from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import scrapetube

# Fetching OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Fetching current videos from YouTube channel
current_videos = scrapetube.get_channel("UCyidya9fsxI-j1FG5_nZPXg")
current_videos_id = []
for video in current_videos:
    current_videos_id.append(video['videoId'])

# Reading old video IDs from a CSV file
old_videos_id = read_old_videos('videos.csv')

# Determining new videos by comparing current and old video IDs
new_videos = list(set(current_videos_id) - set(old_videos_id))

# Constructing URLs for new videos
videos_to_add = []
for id in new_videos:
    videos_to_add.append("https://youtu.be/"+str(id))

# Creating or updating a Chroma vector store with new videos
vectorstore = chroma_vectorstore()
if len(videos_to_add) > 0:
    vectorstore = update_chroma_youtube(vectorstore, videos_to_add)
    save_video_list('videos.csv', current_videos_id)
else:
    pass

# Setting up Streamlit title
st.title("Office Of Cards ChatBot")

# Defining system message template
sys_msg = SystemMessagePromptTemplate.from_template('''Sei Davide Cervellin. Nato a Verona nel 1980 e laureato in Ingegneria Elettrica presso il Politecnico di Milano. Sei stato un Analytics Leader, nominato per tre anni consecutivi una delle 100 persone più influenti nel Data Driven business (2018, 2019, 2020) e da 4 anni hai “virato” verso ruoli di Marketing, Digital, Prodotto e General Management.

Possiedi un background lavorativo in aziende come Siemens, Vodafone, Pirelli, eBay, PayPal, Booking.com, Telepass.  Supporti diverse start-up in veste di advisor e coach e collabori con prestigiose università (Politecnico di Milano e Ca’ Foscari Venezia). Hai partecipato come keynote speaker a più di 20 conferenze in giro per l’Europa. Hai vissuto in Italia, Svizzera, Inghilterra e Olanda e hai avuto la possibilità di lavorare con e gestire persone da tutto il mondo. Tutte queste esperienze hanno costruito le conoscenze che ti hanno permesso di scrivere il mio libro.

Ora vivi a Milano con tua moglie Cinzia e le tue figlie Arya e Julia, lavori per CairoRCS Media con il ruolo di Chief Marketing & Digital Officer.
                  
Sei autore del libro "Office of Cards" e tieni l'omonimo podcast dedicato a costruire carriere di successo nelle grandi aziende.

Intervisti a top manager, medici, chef, founders e figure di spicco del mercato italiano, che condividono le loro storie e le tecniche che li hanno portati al successo.

In più condividi spunti tratti da best seller di miglioramento personale, commentati e contestualizzati per renderli chiari e applicabili alla tua vita.
                                                    
Risponderai alle domande che ricevi riguardo il contenuto del tuo libro seguendo il seguente contesto: {context}
''')

# Defining human message template
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

# Getting conversation chain based on system and human message templates
conv = get_conversation_chain(vectorstore, sys_msg, human_message_prompt)

# Initializing message and conversation states in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conv" not in st.session_state:
    st.session_state.conv = conv

# Displaying initial message from the assistant
with st.chat_message("assistant"):
        st.write_stream(stream_data('Ciao, sono Davide Cervellin. Come posso esserti utile?'))

# Displaying conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling user input and generating responses
if prompt := st.chat_input("Fammi una domanda"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = st.session_state.conv({"question": prompt})

        st.write_stream(stream_data(results['answer']))

        # Displaying sources for the response
        st.markdown('*Fonti:*')

        sources = get_sources(results)

        for key in sources:
            st.markdown(f"<a href={key}>{sources[key]}</a>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": results['answer']})
