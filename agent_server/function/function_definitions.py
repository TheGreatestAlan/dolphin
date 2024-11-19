from agent_server.function.functionRegistry import function_registry


def generate_json_definitions():
    json_definitions = {'available_actions': []}
    for name, info in function_registry.items():
        action_definition = {
            'action': name,
            'parameters': info['parameters'],
            'description': info['description'],
            'examples': info['examples']
        }
        json_definitions['available_actions'].append(action_definition)
    return json_definitions
