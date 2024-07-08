import os, sys
from typing import Any, Dict, List, Optional, Union
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
#from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from credential_utilties.environment import setDataEnvironmentVariablesFromConfig, setAzureOpenaiEnvironmentVariablesFromConfig
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrap_common import llm_wrapper
from okahu_apptrace.wrapper import WrapperMethod
from okahu_apptrace.instrumentor import setup_okahu_telemetry
#from credential_utilties.environment import setOkahuEnvironmentVariablesFromConfig


def init(config_path:str):
    setDataEnvironmentVariablesFromConfig(config_path)
    setAzureOpenaiEnvironmentVariablesFromConfig(config_path)
    setup_okahu_telemetry(
    workflow_name="azure_openai_llama_index_1"
    ,span_processors=[BatchSpanProcessor(ConsoleSpanExporter())]
    ,wrapper_methods=[
        WrapperMethod(
            package="llama_index.llms.openai.base",
            object="OpenAI",
            span_name="llamaindex.azure_openai",
            wrapper=llm_wrapper, 
            method="chat"),
        ]    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def setup_embedding(chroma_vector_store: ChromaVectorStore, embed_model):
    # Creating a Chroma client
    documents = SimpleDirectoryReader(input_files=
                             [os.environ["AZUREML_MODEL_DIR"] + "/coffee_llama_embedding/coffee.txt"]).load_data()

    storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    index.storage_context.persist(persist_dir=os.environ["AZUREML_MODEL_DIR"])

def get_vector_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=os.environ["AZUREML_MODEL_DIR"] + "/coffee_llama_embedding")
    create_embedding = False
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    try:
        chroma_collection = chroma_client.get_collection("quickstart")
    except ValueError:
        chroma_collection = chroma_client.create_collection("quickstart")
        create_embedding = True
    # construct vector store
    chroma_vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
    )
    if create_embedding == True:
        setup_embedding(chroma_vector_store, embed_model)
    return VectorStoreIndex.from_vector_store(vector_store=chroma_vector_store, embed_model=embed_model)

def run():
    az_llm = AzureOpenAI(deployment_id=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
                     api_key=os.environ.get("AZURE_OPENAI_API_KEY"), 
                     api_version=os.environ.get("AZURE_OPENAI_API_VERSION"), 
                     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    index = get_vector_index()
    query_engine = index.as_query_engine(llm=az_llm)

    while True:
        prompt = input("\nAsk a coffee question [Press return to exit]: ")
        if prompt == "":
            break
        response = query_engine.query(prompt)
        print(response)

def main():
    init(sys.argv[1])
    run()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage " + sys.argv[0] + " <config-file-path>")
        sys.exit(1)
    main()

