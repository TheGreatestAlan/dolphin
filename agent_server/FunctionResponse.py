from enum import Enum


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class FunctionResponse:
    def __init__(self, status: Status, response: str):
        self.status = status
        self.response = response

    def to_dict(self):
        return {
            "status": self.status,
            "response": self.response
        }
