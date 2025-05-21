from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter


# Load the PDF file
loader = WebBaseLoader("https://www.allaboutvlsi.com/apb-protocol")
# Load the documents
docs = loader.load()
# Print the number of documents loaded
print(len(docs))

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
)

result = splitter.split_documents(docs)
# Print the number of chunks created
print(len(result))
print(result[1].page_content)
print(result[1].metadata)
