
# Import required modules from the LangChain package
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

#
#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 1.0.0 or higher.
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = "https://team43-openai.openai.azure.com/", 
  #api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_key="01fe36704035437aa74cfbf4c4d86124",
  api_version="2024-02-15-preview"
)


message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."}]

completion = client.chat.completions.create(
  model="team43-deployment", # model = "deployment_name"
  messages = message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)
#

# Load a PDF document and split it into sections
loader = PyPDFLoader("../test1.pdf")
docs = loader.load_and_split()

# Initialize the OpenAI chat model
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.8)

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load the Chroma database from disk
chroma_db = Chroma(persist_directory="data", 
                   embedding_function=embeddings,
                   collection_name="lc_chroma_demo")

# Get the collection from the Chroma database
collection = chroma_db.get()

# If the collection is empty, create a new one
if len(collection['ids']) == 0:
    # Create a new Chroma database from the documents
    chroma_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="data",
        collection_name="lc_chroma_demo"
    )

    # Save the Chroma database to disk
    chroma_db.persist()

# Prepare query
query = "What is this document about?"

print('Similarity search:')
print(chroma_db.similarity_search(query))

print('Similarity search with score:')
print(chroma_db.similarity_search_with_score(query))

# Add a custom metadata tag to the first document
docs[0].metadata = {
    "tag": "demo",
}

# Update the document in the collection
chroma_db.update_document(
    document=docs[0],
    document_id=collection['ids'][0]
)

# Find the document with the custom metadata tag
collection = chroma_db.get(where={"tag" : "demo"})

# Prompt the model
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=chroma_db.as_retriever())

# Execute the chain
response = chain(query)

# Print the response
print(response['result'])

# Delete the collection
chroma_db.delete_collection()