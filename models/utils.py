import random
from llama_index import SimpleDirectoryReader

def load_documents(input_files: list) -> SimpleDirectoryReader:
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    print("Loaded {} documents".format(len(documents)))

    # Shuffle the documents
    random.seed(42)
    random.shuffle(documents)

    return documents
