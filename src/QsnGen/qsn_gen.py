from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import Document
from typing import List

class Qgen:
    def __init__(self, llm, documents: List[Document]):
        self.llm = llm
        self.documents = documents
        self.data_generator = RagDatasetGenerator.from_documents(llm=self.llm, documents=self.documents,num_questions_per_chunk=2)

    def generate_eval_dataset(self,**kwargs):
        rag_dataset = self.data_generator.generate_dataset_from_nodes(**kwargs)
        return rag_dataset
