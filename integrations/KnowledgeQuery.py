from FunctionResponse import FunctionResponse, Status
from llms.LLMInterface import LLMInterface


class KnowledgeQuery:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def query(self, prompt) -> FunctionResponse:
        system_message = "You are a knowledgeable assistant that provides factual information based on a Google-like " \
                         "text knowledge store."
        try:
            return FunctionResponse(Status.SUCCESS, self.llm.generate_response(prompt, system_message))
        except Exception as e:
            return FunctionResponse(Status.FAILURE, e)
