from llama_index.core.evaluation import FaithfulnessEvaluator
import nest_asyncio

nest_asyncio.apply()

class FaithfulnessScore:
    def __init__(self, llm):
        self.llm = llm
        self.eval_model = FaithfulnessEvaluator(llm=llm)
        self.counter = 0
        self.running_total = 0
    def get_score(self,query,response,contexts):
        eval_results = self.eval_model.evaluate(query=query,response=response,contexts=contexts)
        self.counter = self.counter + 1
        result = 1 if eval_results.passing else 0
        self.running_total = self.running_total + result

    def stats(self):
        # Calculate the mean score (percentage of passing responses)
        if self.counter == 0:
            return {
                "total_evaluations": 0,
                "mean_score": 0.0,
                "passing_rate": 0.0,
                "failing_rate": 0.0
            }

        mean_score = self.running_total / self.counter
        passing_rate = self.running_total / self.counter
        failing_rate = 1 - passing_rate

        return {
            "total_evaluations": self.counter,
            "mean_score": mean_score,
            "passing_rate": passing_rate,
            "failing_rate": failing_rate
        }