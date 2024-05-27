from abc import ABC, abstractmethod

class Inventory(ABC):
    @abstractmethod
    def get_inventory(self):
        pass

    @abstractmethod
    def find_location(self, item_name: str):
        pass

    @abstractmethod
    def find_container(self, container_id: str):
        pass

    @abstractmethod
    def create_items(self, container: str, items: list):
        pass

    @abstractmethod
    def delete_items(self, container: str, items: list):
        pass
