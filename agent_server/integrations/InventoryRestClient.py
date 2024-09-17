import json

import requests

from agent_server.FunctionResponse import FunctionResponse, Status
from agent_server.integrations.Inventory import Inventory


class InventoryClient(Inventory):
    def __init__(self, base_url):
        self.base_url = base_url

    def map_to_function_response(self, response):
        if response is None:
            return FunctionResponse(Status.FAILURE, "An error occurred")
        elif response.status_code >= 200 and response.status_code < 300:
            # Successful response (may include 204 No Content)
            try:
                data = response.json()
                if isinstance(data, dict):
                    data_string = json.dumps(data)  # Convert dictionary to JSON string
                else:
                    data_string = str(data)  # Convert other types to string if needed

                return FunctionResponse(Status.SUCCESS, data_string)
            except ValueError:
                # Likely an empty response body
                return FunctionResponse(Status.SUCCESSS, "Action completed successfully")
        else:
            # Handle error responses
            error_msg = f"Error {response.status_code}: {response.text}"
            return FunctionResponse(Status.FAILURE, error_msg)

    def get_inventory(self) -> FunctionResponse:
        url = f"{self.base_url}/inventory"
        return self._make_request('GET', url)

    def find_location(self, item_name) -> FunctionResponse:
        url = f"{self.base_url}/inventory/item/{item_name}"
        return self._make_request('GET', url)

    def get_container(self, container_id)  -> FunctionResponse:
        url = f"{self.base_url}/inventory/item/container/{container_id}"
        return self._make_request('GET', url)

    def create_items(self, container, items) -> FunctionResponse:
        url = f"{self.base_url}/inventory/items"
        data = {"container": container, "items": items}
        return self._make_request('POST', url, json=data)

    def delete_items(self, container, items) -> FunctionResponse:
        url = f"{self.base_url}/inventory/items"
        data = {"container": container, "items": items}
        return self._make_request('DELETE', url, json=data)

    def _make_request(self, method, url, **kwargs) -> FunctionResponse:
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()  # Raise for HTTP errors (200-299)
            return self.map_to_function_response(response)
        except requests.exceptions.RequestException as err:  # Catch all request errors
            error_msg = f"Request error: {err}"
            return FunctionResponse(Status.FAILURE, error_msg)


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
