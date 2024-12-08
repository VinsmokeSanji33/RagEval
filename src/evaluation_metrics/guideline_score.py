
from llama_index.core.evaluation import GuidelineEvaluator
import nest_asyncio

nest_asyncio.apply()
class GuidelineScore:
    def __init__(self, llm , guidelines):
        self.llm = llm
        self.guidelines = guidelines
        self.evaluators  = [GuidelineEvaluator(llm=llm, guidelines=guide) for guide in self.guidelines]
        self.counterList = [0 for i in range(len(guidelines))]
        self.running_total_List = [0 for i in range(len(guidelines))]

    def get_score(self,query,response,contexts,indexs):
        for idx , (guideline, evaluator) in enumerate(zip(self.guidelines, self.evaluators)):
            if idx in indexs:
                eval_result = evaluator.evaluate(
                    query=query,
                    contexts=contexts,
                    response=response,
                )
                result = 1 if eval_result.passing else 0
                self.counterList[idx] = self.counterList[idx] + 1
                self.running_total_List[idx] = self.running_total_List[idx] + result
            
    def stats(self):
        # Calculate total evaluations
        total_evaluations = sum(self.counterList)
        if total_evaluations == 0:
            return {
                "total_evaluations": 0,
                "mean_score": 0.0,
                "passing_rate": 0.0,
                "failing_rate": 0.0,
                "index_wise_passing_rate": []
            }

        total_passes = sum(self.running_total_List)
        mean_score = total_passes / total_evaluations
        passing_rate = mean_score
        failing_rate = 1 - passing_rate

        # Calculate index-wise passing rate
        index_wise_passing_rate = []
        for idx in range(len(self.guidelines)):
            if self.counterList[idx] == 0:
                rate = 0.0
            else:
                rate = self.running_total_List[idx] / self.counterList[idx]
            index_wise_passing_rate.append({
                "guideline_index": idx,
                "passing_rate": rate
            })

        return {
            "total_evaluations": total_evaluations,
            "mean_score": mean_score,
            "passing_rate": passing_rate,
            "failing_rate": failing_rate,
            "index_wise_passing_rate": index_wise_passing_rate
        }