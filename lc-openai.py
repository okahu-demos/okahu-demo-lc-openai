import os, sys
import warnings, logging 
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.errors import InvalidCollectionException
from langchain import hub
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader
#from okahu_apptrace.instrumentor import setup_okahu_telemetry

def setup_embedding(chroma_vector_store: Chroma ):
    print("Creating vector store")
    documents = DirectoryLoader("data/coffee_llama_embedding", glob="*.txt").load()
    chroma_vector_store.add_documents(documents)

def get_vector_store() :
    chroma_client = chromadb.PersistentClient("data/coffee_llama_embedding")
    create_embedding = False
    chroma_collection_name = "okahu-demo"
#    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        chroma_collection = chroma_client.get_collection(chroma_collection_name)
    except InvalidCollectionException:
        chroma_collection = chroma_client.create_collection(chroma_collection_name)
        create_embedding = True
    # construct vector store
    chroma_vector_store = Chroma(
        collection_name=chroma_collection_name, embedding_function=embed_model
    )
    if create_embedding == True:
        setup_embedding(chroma_vector_store )
    return chroma_vector_store

def run():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    def format_docs(docs):
        return "\n\n ".join(doc.page_content for doc in docs)

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        prompt = input("\nAsk a coffee question [Press return to exit]: ")
        if prompt == "":
            break
        response = rag_chain.invoke(prompt)
        print(response)

def main():
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    run()

if __name__ == "__main__":
    main()

