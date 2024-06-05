# Importing necessary modules and classes
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

# Function to update the Chroma vector store with new YouTube videos
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

# Function to create or load the Chroma vector store
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

# Function to create a conversation chain for the chatbot
def get_conversation_chain(vector_store, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0) 
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

# Function to retrieve a list of video URLs from a given source (channel or playlist)
def get_video_list(source) -> list:
    videos = scrapetube.get_playlist(source)
    video_list = []

    for video in videos:
        video_list.append("https://youtu.be/"+str(video['videoId']))
    
    return video_list

# Function to stream data
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# Function to extract sources from the results of a query
def get_sources(results):
    sources = {}
    for source in results['source_documents']:
        if 'title' in source.metadata.keys():
            text = source.metadata['title']
        else:
            text = 'Office of Cards: Una guida pratica per raggiungere il successo e la felicità nelle grandi aziende (e nella vita)'
        
        sources[source.metadata['url']] = text

    return sources

# Function to read old video IDs from a CSV file
def read_old_videos(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data[0]

# Function to save video IDs to a CSV file
def save_video_list(name, video_exp):
     with open(name, 'w', newline='') as myfile:
          wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
          wr.writerow(video_exp)

# Check if there are new videos in the channel and update the vectorstore
def check_and_update_new_videos(channel_id, old_videos_file, vectorstore):
    # Fetching current videos from YouTube channel
    current_videos = scrapetube.get_channel(channel_id)
    current_videos_id = [video['videoId'] for video in current_videos]

    old_videos_id = read_old_videos(old_videos_file)

    # Constructing URLs for new videos
    videos_to_add = ["https://youtu.be/"+str(id) for id in list(set(current_videos_id) - set(old_videos_id))]

    # Creating or updating a Chroma vector store with new videos
    if len(videos_to_add) > 0:
        vectorstore = update_chroma_youtube(vectorstore, videos_to_add)
        save_video_list(old_videos_file, current_videos_id)
    else:
        pass

    return vectorstore
