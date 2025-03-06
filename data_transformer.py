from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

loader = PyPDFLoader("attention.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
final_document = text_splitter.split_documents(docs)
# print(final_document[0].page_content, "---", final_document[1].page_content)

text_loader = TextLoader("speech.txt")
text_load = text_loader.load()
# print(text_load)

web_loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")
                     )))
web_load = web_loader.load()
# print(web_load)

arxivLoader = ArxivLoader(query = "Astrophysics", load_max_docs = 2).load()
# print(arxivLoader, arxivLoader[0].metadata)

wikipedia_loader = WikipediaLoader(query= "What is Astrophysics", load_max_docs= 2).load()
print(wikipedia_loader)
