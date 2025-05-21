from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",  # should be a folder with pdf files
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
