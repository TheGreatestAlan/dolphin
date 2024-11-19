import os

from agent_server.function.FunctionMapper import FunctionMapper
from agent_server.InventoryFunctionGenerator import InventoryFunctionGenerator
from agent_server.agent.Sequencer import Sequencer
from agent_server.agent.JsonFunctionCreator import JsonFunctionCreator
from agent_server.integrations.InventoryRestClient import InventoryClient
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery
from agent_server.integrations.SmartFindingInventoryClient import SmartFindingInventoryClient
from agent_server.integrations.local_device_action import LocalDeviceAction
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.llms.LLMInterface import LLMInterface

class TaskOrchestrator:
    def __init__(self, llm_interface: LLMInterface):
        self.sequencer = Sequencer(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_2_11B))
        self.json_function_creator = JsonFunctionCreator(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_8B))

        self.chat_handler = None
        rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
        smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, llm_interface)
        function_generator = InventoryFunctionGenerator(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_8B))
        knowledge_query = KnowledgeQuery(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_8B))
        local_device_action = LocalDeviceAction(None)
        self.function_mapper = FunctionMapper(smart_finding_inventory_client, function_generator, self.chat_handler,
                                              knowledge_query, local_device_action)

    def process_user_request(self, user_request: str) -> list:
        tasks = self.sequencer.parse_tasks(user_request)
        results = []

        for task in tasks:  # Assuming 'tasks' is your JSON list
            task_description = task.get("task")
            chosen_function = task.get("function")

            if task_description and chosen_function:
                json_output = self.json_function_creator.create_json(chosen_function, task_description)
                function_response = self.function_mapper.handle_function_call(json_output, None)
                print(function_response)
            else:
                print("Error: Missing 'task' or 'function' in task:", task)
        return results

    def shutdown(self):
        """Included for interface compatibility, but does nothing as there's no parallel processing."""
        pass
