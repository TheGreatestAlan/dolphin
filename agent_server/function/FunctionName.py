import os
from enum import Enum

DEFINITIONS_DIR = os.path.join(os.path.dirname(__file__), "definition")

class FunctionName(Enum):
    GET_CONTAINER = ("get_container", "get_container.txt")
    CREATE_ITEMS = ("create_items", "create_items.txt")
    DELETE_ITEMS = ("delete_items", "delete_items.txt")
    GET_INVENTORY = ("get_inventory", "get_inventory.txt")
    FIND_LOCATION = ("find_location", "find_location.txt")
    KNOWLEDGE_QUERY = ("knowledge_query", "knowledge_query.txt")
    EVENT_ALERT_ACTION = ("event_alert_action", "event_alert_action.txt")

    @property
    def function_name(self):
        return self.value[0]

    @property
    def definition_path(self):
        # Join the base directory path with the filename to get the full path
        return os.path.join(DEFINITIONS_DIR, self.value[1])

    @classmethod
    def has_value(cls, value):
        return value in [item.function_name for item in cls]
