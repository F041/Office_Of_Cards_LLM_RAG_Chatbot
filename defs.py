from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_community.document_loaders import YoutubeLoader, DirectoryLoader, UnstructuredEPubLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
)
import time
import scrapetube
import csv

def update_chroma_youtube(vectorstore, video_list):
    for url in video_list:
        loader_yt = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["it"],
            )
        documents = loader_yt.load()
        documents[0].metadata['url'] = url
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)

        vectorstore.add_documents(docs)
    
    return vectorstore

def chroma_vectorstore(video_list = []):
    save_load = './vectorstore/chroma'
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    if os.path.exists(save_load):
        vector_store = Chroma(persist_directory=save_load, embedding_function=embeddings)
        print("A new instance of vector store was loaded from", save_load)
    else:
        loader_epub = DirectoryLoader('./sources/', glob="./*.epub", loader_cls=UnstructuredEPubLoader)
        documents = loader_epub.load()
        documents[0].metadata['url'] = 'https://amzn.to/3UX6umg'
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=25)
        text_splitter = NLTKTextSplitter()
        docs = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=save_load
        )

        print("Vector store was saved to", save_load)

        if len(video_list) == 0:
            pass
        else:
            return update_chroma_youtube(vector_store, video_list)
            
    return vector_store

def get_conversation_chain(vector_store, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Oggetto LangChain che permette domanda-risposta tra umano e LLM

    Args:
        vector_store: Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0) # possiamo cambiare modello a piacimento
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type = 'mmr', search_kwargs = {'k' : 5} ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def get_video_list(source) -> list:
    """
    Retrieve a video url list from a channel or youtube playlist.

    Args:
        source: channel or playlist ID

    Returns:
        video_list: unordered video list
    """

    videos = scrapetube.get_playlist(source)
    video_list = []

    for video in videos:
        video_list.append("https://youtu.be/"+str(video['videoId']))
    
    return video_list

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def get_sources(results):
    sources = {}
    for source in results['source_documents']:
        if 'title' in source.metadata.keys():
            text = source.metadata['title']
        else:
            text = 'Office of Cards: Una guida pratica per raggiungere il successo e la felicità nelle grandi aziende (e nella vita)'
        
        sources[source.metadata['url']] = text

    return sources

def read_old_videos(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data[0]