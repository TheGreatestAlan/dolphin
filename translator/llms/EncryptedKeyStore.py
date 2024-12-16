import os
import json
from cryptography.fernet import Fernet


class EncryptedKeyStore:
    def __init__(self, encrypted_file_name="keys.json.enc"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.encrypted_file_path = os.path.join(script_dir, "../dev", encrypted_file_name)

        # Retrieve encryption key from environment
        self.encryption_key = os.getenv("ENCRYPTION_PASSWORD")
        if not self.encryption_key:
            raise EnvironmentError("Encryption password not set in environment variables.")

        self.fernet = Fernet(self.encryption_key)

        # Check if the encrypted file exists; if not, create it
        if not os.path.exists(self.encrypted_file_path):
            print(f"Encrypted file {self.encrypted_file_path} not found. Creating a new one.")
            self._create_default_encrypted_file()

    def _create_default_encrypted_file(self):
        # Default data structure for keys; modify as needed
        default_data = {
            "CHATGPT4_API_KEY": "default_chatgpt4_key",
            "FIREWORKS_API_KEY": "default_fireworks_key",
            "OLLAMA_API_KEY": "default_ollama_key"
        }

        # Encrypt and save the default data
        self._encrypt_file(default_data)
        print(f"New encrypted file created at {self.encrypted_file_path} with default values.")

    def _decrypt_file(self):
        with open(self.encrypted_file_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
        decrypted_data = self.fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data)

    def _encrypt_file(self, data):
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        with open(self.encrypted_file_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)

    def get_api_key(self, key_name):
        data = self._decrypt_file()
        return data.get(key_name)

    def add_api_key(self, key_name, key_value, overwrite=False):
        """
        Adds a new API key to the encrypted file.
        If the key already exists and overwrite is False, it raises an error.
        If overwrite is True, it will replace the existing key value.
        """
        data = self._decrypt_file()

        if key_name in data and not overwrite:
            print(f"Key '{key_name}' already exists. Use overwrite=True to replace it.")
            return

        # Add or update the key
        data[key_name] = key_value
        self._encrypt_file(data)
        if key_name in data:
            print(f"Key '{key_name}' updated successfully.")
        else:
            print(f"Key '{key_name}' added successfully.")


# Example usage
if __name__ == "__main__":
    key_store = EncryptedKeyStore()
    print("Initialized EncryptedKeyStore and checked for encrypted file.")




# Example usage
if __name__ == "__main__":
    key_store = EncryptedKeyStore()
    key_store.add_api_key("NEW_API_KEY", "new_api_key_value")  # Add a new key
    key_store.add_api_key("CHATGPT4_API_KEY", "updated_chatgpt4_key_value", overwrite=True)  # Overwrite an existing key
