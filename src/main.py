import os
from evaluation_metrics.correctness_score import CorrectnessScore
from evaluation_metrics.faithfulness_score import FaithfulnessScore
from evaluation_metrics.guideline_score import GuidelineScore
from evaluation_metrics.relevancy_score import RelevancyScore
# from rag_evaluation.graphrag import Graphrag
# from rag_evaluation.lightrag import Lightrag
from rag_evaluation.navierag import Naiverag
# from rag_evaluation.vector_sql_rag import SQLrag
from QsnGen.qsn_gen import Qgen
from tools.utils import utils
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import nest_asyncio
import random
from dotenv import load_dotenv
from llama_index.core.llama_dataset import LabelledRagDataExample
load_dotenv()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# loads BAAI/bge-small-en
# embed_model = HuggingFaceEmbedding()

# loads BAAI/bge-small-en-v1.5

nest_asyncio.apply()

reader = SimpleDirectoryReader("/home/rohit/RagEval/data/")
documents = reader.load_data(show_progress=True)

llm = OpenAI(model="gpt-4",api_key=os.getenv("api-key"))
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

dataset_generator = Qgen(
    documents=documents,
    llm=llm, # set the number of questions per nodes
)

rag_dataset = dataset_generator.generate_eval_dataset()

gudielines = utils(llm).generate_gudielines()

correctnessData = CorrectnessScore(llm)
FaithfulnessData = FaithfulnessScore(llm)
RelevancyData = RelevancyScore(llm)
GuidelineData = GuidelineScore(llm,gudielines)
naive_rag_engine = Naiverag(documents=documents,llm=llm,embed_model=embed_model)
print(type(rag_dataset))
for question in rag_dataset.examples:
    #TODO pass the question to the rag modules and store the answers
    question = LabelledRagDataExample.parse_raw(question.json())
    question = question.dict()
    print(question["reference_answer"])
    query = question["query"]
    reference_contexts = question["reference_contexts"]
    reference_contexts_str = str(question["reference_contexts"])
    reference_answer = question["reference_answer"]
    # Navie Rag
    naive_rag_response = naive_rag_engine.query(query) 
    print(type(naive_rag_response))
    print("naive rag response")
    print(naive_rag_response)
    naive_rag_answer = str(naive_rag_response)
    print(type(naive_rag_answer))
    print("naive rag answer")
    print(naive_rag_answer)
    naive_rag_source = naive_rag_response.source_nodes[0].node.text[:1000] + "..."
    print(type(naive_rag_source))
    print("naive rag source")
    print(naive_rag_response)
    # Graph Rag
    # Light Rag
    # SQL Rag 
    guideline_index = random.randint(0,len(gudielines)-1)
    correctnessData.get_score(query,reference_contexts_str,reference_answer)
    FaithfulnessData.get_score(query,reference_answer,reference_contexts)
    RelevancyData.get_score(query,reference_answer,reference_contexts)
    GuidelineData.get_score(query,reference_answer,reference_contexts,[guideline_index])
m1 = correctnessData.stats()
m2 = FaithfulnessData.stats()
m3 = RelevancyData.stats()
m4 = GuidelineData.stats()
print(m1)




    