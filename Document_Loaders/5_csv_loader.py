from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv")

docs = loader.load()

print(len(docs))
print(docs[1])


"""
400
Purchased: 0' metadata={'source': 'Social_Network_Ads.csv', 'row': 1}    
"""
