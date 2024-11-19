# dependencies.py
from agent_server.integrations.ChatHandler import ChatHandler
from agent_server.integrations.Inventory import Inventory
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery
from agent_server.integrations.local_device_action import LocalDeviceAction

dependencies = {
    'inventory': Inventory(),
    'chat_handler': ChatHandler(),
    'knowledge_query': KnowledgeQuery(),
    'local_device_action': LocalDeviceAction(),
    # Add other dependencies as needed
}
