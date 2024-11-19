
from llama_index.core.evaluation import CorrectnessEvaluator
import nest_asyncio

nest_asyncio.apply()
class CorrectnessScore:
    def __init__(self, llm):
        self.llm = llm
        self.eval_model = CorrectnessEvaluator(llm=llm)
        self.counter = 0
        self.running_total = 0
    def get_score(self,query,response,reference):
        eval_results = self.eval_model.evaluate(query=query,response=response,reference=reference)
        self.counter = self.counter + 1
        score = eval_results.score
        self.running_total = self.running_total + score

    def stats(self):
        # Calculate the mean score (percentage of passing responses)
        if self.counter == 0:
            return {
                "total_evaluations": 0,
                "mean_score": 0.0,
            }

        mean_score = self.running_total / self.counter

        return {
            "total_evaluations": self.counter,
            "mean_score": mean_score,
        }