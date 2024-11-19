# function_mapper.py
from agent_server.FunctionResponse import FunctionResponse, Status
from agent_server.function.functionRegistry import function_registry


class FunctionMapper:
    def __init__(self):
        pass  # Initialization if needed

    def wrap_to_action_response(self, function_response, action_name):
        status = function_response.status.name if hasattr(function_response, 'status') else 'SUCCESS'
        value = function_response.response if hasattr(function_response, 'response') else function_response
        return {
            'action_name': action_name,
            'status': status,
            'value': value
        }

    def handle_function_call(self, prompt: dict, session_id: str):
        action = prompt.get('action')
        parameters = prompt.get('parameters', {})

        if not action:
            return {'action_name': None, 'response': "Invalid function call format: missing 'action'"}

        if action not in function_registry:
            return {
                'action_name': action,
                'response': f"Unknown action: {action}. Expected actions: {list(function_registry.keys())}"
            }

        function_info = function_registry[action]
        function = function_info['function']
        required_params = function_info['parameters']

        # Check for missing parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return {
                'action_name': action,
                'response': f"Missing required parameter(s) {missing_params} for action '{action}'."
            }

        # Prepare arguments
        kwargs = {param: parameters[param] for param in required_params}

        # Include session_id if required by the function
        if 'session_id' in function.__code__.co_varnames:
            kwargs['session_id'] = session_id

        # Execute the function
        try:
            result = function(**kwargs)
            return self.wrap_to_action_response(result, action)
        except Exception as e:
            print(f"Error executing function '{action}': {e}")
            return {
                'action_name': action,
                'response': f"Error executing function '{action}': {str(e)}"
            }

    def list_functions(self):
        from function_definitions import generate_json_definitions
        return FunctionResponse(Status.SUCCESS, generate_json_definitions())
