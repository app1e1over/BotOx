from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

from app.config.env import get_settings
import os
env = get_settings()
os.environ['OPENAI_API_KEY'] = env.OPENAI_API_KEY

# Initialize the index
documents = SimpleDirectoryReader('./documents').load_data()


# Convert text to a document
try:
    index = GPTVectorStoreIndex().from_documents(documents)
except Exception as e:
    print(e)
    
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

documents = SimpleDirectoryReader('./documents').load_data()

index = GPTVectorStoreIndex.from_documents(documents)

index.storage_context.persist('storage')