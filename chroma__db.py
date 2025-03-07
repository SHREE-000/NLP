from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = TextLoader("speech.txt")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embedding=OllamaEmbeddings(model="gemma2:2b")
vectordb=Chroma.from_documents(documents=splits,embedding=embedding)
query = "What does the speaker believe is the main reason the United States should enter the war?"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)
vectordb=Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db")
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
docs=db2.similarity_search(query)
print(docs[0].page_content)