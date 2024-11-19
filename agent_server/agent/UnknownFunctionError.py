class UnknownFunctionError(Exception):
    """Exception raised when an unknown function name is provided."""
    def __init__(self, function_name):
        self.function_name = function_name
        super().__init__(f"Unknown function name: {function_name}")
