from abc import ABC, abstractmethod

from FunctionResponse import FunctionResponse


class Inventory(ABC):
    @abstractmethod
    def get_inventory(self) -> FunctionResponse:
        pass

    @abstractmethod
    def find_location(self, item_name: str) -> FunctionResponse:
        pass

    @abstractmethod
    def get_container(self, container_id: str) -> FunctionResponse:
        pass

    @abstractmethod
    def create_items(self, container: str, items: list) -> FunctionResponse:
        pass

    @abstractmethod
    def delete_items(self, container: str, items: list) -> FunctionResponse:
        pass
