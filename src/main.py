from evaluation_metrics.correctness_score import CorrectnessScore
from evaluation_metrics.faithfulness_score import FaithfulnessScore
from evaluation_metrics.guideline_score import GuidelineScore
from evaluation_metrics.relevancy_score import RelevancyScore
from QsnGen.qsn_gen import Qgen
from tools.utils import utils
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import nest_asyncio
import random

from llama_index.core.llama_dataset import LabelledRagDataExample


nest_asyncio.apply()

reader = SimpleDirectoryReader("/home/rohit/RagEval/data/")
documents = reader.load_data(show_progress=True)

llm = OpenAI(model="gpt-4",api_key="api-key")

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

print(type(rag_dataset))
for question in rag_dataset.examples:
    print(1)
    question = LabelledRagDataExample.parse_raw(question.json())
    question = question.dict()
    print(question["reference_answer"])
    query = question["query"]
    reference_contexts = question["reference_contexts"]
    reference_contexts_str = str(question["reference_contexts"])
    reference_answer = question["reference_answer"]
    guideline_index = random.randint(0,len(gudielines)-1)
    print(guideline_index)
    correctnessData.get_score(query,reference_contexts_str,reference_answer)
    FaithfulnessData.get_score(query,reference_answer,reference_contexts)
    RelevancyData.get_score(query,reference_answer,reference_contexts)
    GuidelineData.get_score(query,reference_answer,reference_contexts,[guideline_index])
m1 = correctnessData.stats()
m2 = FaithfulnessData.stats()
m3 = RelevancyData.stats()
m4 = GuidelineData.stats()
print(m1)




    