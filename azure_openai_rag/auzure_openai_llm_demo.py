import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrap_common import llm_wrapper
from okahu_apptrace.wrapper import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


setup_okahu_telemetry(
    workflow_name="llama_index_1",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[
        WrapperMethod(
            package="llama_index.llms.openai.base",
            object="OpenAI",
            method="chat",
            span_name="llamaindex.openai",
            wrapper=llm_wrapper),
        ]
)

# Creating a Chroma client
# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)
dir_path = os.path.dirname(os.path.realpath(__file__))
documents = SimpleDirectoryReader(dir_path + "/data").load_data()

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

az_llm = AzureOpenAI(deployment_id="kshitiz-gpt",api_key=os.environ.get("AZURE_OPENAI_API_KEY"), api_version="2024-02-01", azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"))
llm = OpenAI(temperature=0.1, model="gpt-4")

query_engine = index.as_query_engine(llm= llm, )
response = query_engine.query("What did the author do growing up?")

print(response)

