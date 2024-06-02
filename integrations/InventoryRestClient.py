import requests

from integrations.Inventory import Inventory


class InventoryClient(Inventory):
    def __init__(self, base_url):
        self.base_url = base_url

    def get_inventory(self):
        url = f"{self.base_url}/inventory"
        response = self._make_request('GET', url)
        return response

    def find_location(self, item_name):
        url = f"{self.base_url}/inventory/item/{item_name}"
        response = self._make_request('GET', url)
        return response

    def get_container(self, container_id):
        url = f"{self.base_url}/inventory/item/container/{container_id}"
        response = self._make_request('GET', url)
        return response

    def create_items(self, container, items):
        url = f"{self.base_url}/inventory/items"
        data = {"container": container, "items": items}
        response = self._make_request('POST', url, json=data)
        return response

    def delete_items(self, container, items):
        url = f"{self.base_url}/inventory/items"
        data = {"container": container, "items": items}
        response = self._make_request('DELETE', url, json=data)
        return response

    def _make_request(self, method, url, **kwargs):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            if response.status_code != 204:  # No Content
                return response.json()
            return {"message": "Action completed successfully"}
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        return None


# Example usage:
if __name__ == "__main__":
    base_url = "http://localhost:8080"  # Replace with your actual base URL
    client = InventoryClient(base_url)

    print("Getting inventory:")
    inventory = client.get_inventory()
    print(inventory)

    item_name = "screwdriver"
    print(f"Finding location for item '{item_name}':")
    location = client.find_location(item_name)
    print(location)

    container_id = "15"
    print(f"Finding container with ID '{container_id}':")
    container = client.find_container(container_id)
    print(container)

    new_items = ["hammer", "wrench"]
    print(f"Adding items to container '{container_id}':")
    create_response = client.create_items(container_id, new_items)
    print(create_response)

    items_to_delete = ["hammer"]
    print(f"Deleting items from container '{container_id}':")
    delete_response = client.delete_items(container_id, items_to_delete)
    print(delete_response)
