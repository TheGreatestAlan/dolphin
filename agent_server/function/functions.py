# functions.py
import os

from agent_server.function.decorators import register_function
from agent_server.function.function_definitions import generate_json_definitions
from agent_server.integrations.ChatHandler import ChatHandler
from agent_server.integrations.InventoryRestClient import InventoryClient
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery
from agent_server.integrations.SmartFindingInventoryClient import SmartFindingInventoryClient
from agent_server.integrations.local_device_action import LocalDeviceAction
from agent_server.FunctionResponse import FunctionResponse, Status
from agent_server.llms.LLMFactory import LLMFactory, ModelType


# Assuming you have the necessary imports and initializations for dependencies
# such as inventory, chat_handler, knowledge_query_service, local_device_action

# Initialize dependencies (These should be initialized appropriately in your actual code)
rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
inventory = SmartFindingInventoryClient(rest_inventory_client, LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_2_11B))
chat_handler = ChatHandler()
knowledge_query_service = KnowledgeQuery(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_405B))
local_device_action = LocalDeviceAction(None)

@register_function(
    name='get_inventory',
    parameters={},
    description='Retrieve the entire inventory.',
    examples=[
        {
            'query': 'retrieve the entire inventory',
            'response': {
                'action': 'get_inventory',
                'parameters': {}
            }
        }
    ]
)
def get_inventory():
    return inventory.get_inventory()

@register_function(
    name='find_location',
    parameters={'item_name': 'string'},
    description='Find the location of an item by name.',
    examples=[
        {
            'query': 'find the location of item named screwdriver',
            'response': {
                'action': 'find_location',
                'parameters': {
                    'item_name': 'screwdriver'
                }
            }
        }
    ]
)
def find_location(item_name):
    return inventory.find_location(item_name)

@register_function(
    name='get_container',
    parameters={'container_id': 'string'},
    description='Get details of a specific container.',
    examples=[
        {
            'query': "What's in container 5?",
            'response': {
                'action': 'get_container',
                'parameters': {
                    'container_id': '5'
                }
            }
        }
    ]
)
def get_container(container_id):
    return inventory.get_container(container_id)

@register_function(
    name='create_items',
    parameters={'container': 'string', 'items': ['string']},
    description='Create items in a specified container.',
    examples=[
        {
            'query': 'add a hammer to container 5',
            'response': {
                'action': 'create_items',
                'parameters': {
                    'container': '5',
                    'items': ['hammer']
                }
            }
        },
        {
            'query': 'In container 7 add a screwdriver, gorilla glue, nerf balls, and a fan',
            'response': {
                'action': 'create_items',
                'parameters': {
                    'container': '7',
                    'items': ['screwdriver', 'gorilla glue', 'nerf balls', 'fan']
                }
            }
        }
    ]
)
def create_items(container, items):
    return inventory.create_items(container, items)

@register_function(
    name='delete_items',
    parameters={'container': 'string', 'items': ['string']},
    description='Delete items from a specified container.',
    examples=[
        {
            'query': 'remove a screwdriver from container 10',
            'response': {
                'action': 'delete_items',
                'parameters': {
                    'container': '10',
                    'items': ['screwdriver']
                }
            }
        },
        {
            'query': 'Oops I messed up, delete the phone from container 6',
            'response': {
                'action': 'delete_items',
                'parameters': {
                    'container': '6',
                    'items': ['phone']
                }
            }
        }
    ]
)
def delete_items(container, items):
    return inventory.delete_items(container, items)

@register_function(
    name='send_message',
    parameters={'content': 'string'},
    description='Send a message in a chat session.',
    examples=[
        {
            'query': 'Tell me a joke.',
            'response': {
                'action': 'send_message',
                'parameters': {
                    'content': 'Tell me a joke.'
                }
            }
        }
    ]
)
def send_message(content, session_id):
    return chat_handler.send_message(session_id, content)

@register_function(
    name='list_actions',
    parameters={},
    description='List all available actions.',
    examples=[
        {
            'query': 'What can you do?',
            'response': {
                'action': 'list_actions',
                'parameters': {}
            }
        }
    ]
)
def list_actions():
    return FunctionResponse(Status.SUCCESS, generate_json_definitions())

@register_function(
    name='poll_response',
    parameters={},
    description='Poll for a response in a chat session.',
    examples=[
        {
            'query': 'Check if there is a new message.',
            'response': {
                'action': 'poll_response',
                'parameters': {}
            }
        }
    ]
)
def poll_response(session_id):
    return chat_handler.poll_response(session_id)

@register_function(
    name='start_session',
    parameters={},
    description='Start a new chat session.',
    examples=[
        {
            'query': 'Start a new session.',
            'response': {
                'action': 'start_session',
                'parameters': {}
            }
        }
    ]
)
def start_session():
    return chat_handler.get_or_create_user()

@register_function(
    name='end_session',
    parameters={},
    description='End the current chat session.',
    examples=[
        {
            'query': 'End the session.',
            'response': {
                'action': 'end_session',
                'parameters': {}
            }
        }
    ]
)
def end_session(session_id):
    return chat_handler.end_session(session_id)

@register_function(
    name='knowledge_query',
    parameters={'query': 'string'},
    description='Perform a knowledge query.',
    examples=[
        {
            'query': 'What is the main street in Denver?',
            'response': {
                'action': 'knowledge_query',
                'parameters': {
                    'query': 'What is the main street in Denver?'
                }
            }
        }
    ]
)
def knowledge_query_function(query):
    return knowledge_query_service.query(query)

@register_function(
    name='event_alert_action',
    parameters={'event_name': 'string', 'message': 'string'},
    description='Set an event alert with a message.',
    examples=[
        {
            'query': 'Hey when I get home can you remind me to put in the laundry?',
            'response': {
                'action': 'event_alert_action',
                'parameters': {
                    'event_name': 'arrivedHomeEvent',
                    'message': 'put in laundry'
                }
            }
        }
    ]
)
def event_alert_action(event_name, message, session_id):
    return local_device_action.event_alert_action(event_name, message, session_id)
