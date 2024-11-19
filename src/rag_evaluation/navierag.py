from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    Settings,
)
class Naiverag:
    def __init__(self , llm , documents , vector_store = None , persist_dir=None , embed_model=None , chunk_size = 256 , chunk_overlap = 20 ):
        self.llm = llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.embed_model = embed_model 
        Settings.llm=llm
        if vector_store is None:
            self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        else:
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir=persist_dir)
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine(llm=llm)
    def query(self,question):
        return self.query_engine.query(question)
