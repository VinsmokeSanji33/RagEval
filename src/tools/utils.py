#TODO add more utils for more checks
class utils:
    def __init__(self, llm):
        self.llm = llm
        self.gudielines = []
    def generate_gudielines(self,type_of_guidelines="generic"):
        #TODO ADD domain related guidelines (this can be tackled by the cmplx question generator where we can ask multi level questions)
        if type_of_guidelines is "generic":
            with open('/home/rohit/RagEval/src/tools/guidelines.txt', 'r') as file:
                for line in file:
                    # Strip trailing whitespace and append to the list
                    self.gudielines.append(line.strip())
                    return self.gudielines









