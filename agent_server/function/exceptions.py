# exceptions.py
class FunctionMappingError(Exception):
    pass

class UnknownActionError(FunctionMappingError):
    pass

class ValidationError(FunctionMappingError):
    pass
