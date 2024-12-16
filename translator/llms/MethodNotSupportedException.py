class MethodNotSupportedException(Exception):
    def __init__(self, method_name: str, reason: str):
        super().__init__(f"The method '{method_name}' is not supported: {reason}")
