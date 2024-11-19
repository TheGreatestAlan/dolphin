# decorators.py
from agent_server.function.functionRegistry import function_registry


def register_function(name, parameters=None, description='', examples=None):
    def decorator(func):
        function_registry[name] = {
            'function': func,
            'parameters': parameters,
            'description': description,
            'examples': examples or [],
        }
        return func
    return decorator
