from langchain_community.document_loaders import TextLoader
loader = TextLoader("bad.txt", encoding='utf-8')
documents = loader.load()
print(documents)