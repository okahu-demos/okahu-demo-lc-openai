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

# Create vectore store and load data 
def setup_embedding(embedding_model):
    documents = DirectoryLoader("data/", glob="*.txt").load()
    vector_store = Chroma.from_documents(documents, embedding_model,
        persist_directory="data/vector_store")
    return vector_store

def get_vector_store() :
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(persist_directory="data/vector_store",
            embedding_function=embedding_model)
    if len(vector_store.get()["ids"]) == 0:
        setup_embedding(embedding_model)
    return vector_store

def run(vector_store):
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
        question = input("\nAsk a coffee question [Press return to exit]: ")
        if question == "":
            break
        response = rag_chain.invoke(question)
        print(response)

def main():
    vector_store = get_vector_store()
    run(vector_store)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    main()

