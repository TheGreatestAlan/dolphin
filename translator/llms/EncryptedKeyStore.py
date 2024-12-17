import os
import json
from cryptography.fernet import Fernet


class EncryptedKeyStore:
    def __init__(self, encrypted_file_name="keys.json.enc"):
        # Check for FIREWORKS_API_KEY in environment and bypass encryption if set
        self.env_keys = {
            "FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY"),
            "CHATGPT4_API_KEY": os.getenv("CHATGPT4_API_KEY"),
            "OLLAMA_API_KEY": os.getenv("OLLAMA_API_KEY")
        }

        # If any environment variable exists, bypass encryption logic
        if any(self.env_keys.values()):
            print("Environment keys detected. Skipping encrypted file access.")
            self.use_env_only = True
            return
        else:
            self.use_env_only = False

        # Normal encrypted file setup
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.encrypted_file_path = os.path.join(script_dir, "../dev", encrypted_file_name)

        # Retrieve encryption key from environment
        self.encryption_key = os.getenv("ENCRYPTION_PASSWORD")
        if not self.encryption_key:
            raise ValueError("ENCRYPTION_PASSWORD is not set in the environment.")

        self.fernet = Fernet(self.encryption_key)

        # Check if the encrypted file exists; if not, create it
        if not os.path.exists(self.encrypted_file_path):
            print(f"Encrypted file {self.encrypted_file_path} not found. Creating a new one.")
            self._create_default_encrypted_file()

    def _create_default_encrypted_file(self):
        """Create an encrypted file with default keys."""
        default_data = {
            "CHATGPT4_API_KEY": "default_chatgpt4_key",
            "FIREWORKS_API_KEY": "default_fireworks_key",
            "OLLAMA_API_KEY": "default_ollama_key"
        }
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
        """
        Retrieve the API key. Prefer environment variable if available.
        """
        # Return from environment variable if present
        if self.use_env_only and self.env_keys.get(key_name):
            return self.env_keys.get(key_name)

        # Otherwise, return from the encrypted file
        if not self.use_env_only:
            data = self._decrypt_file()
            return data.get(key_name)

        return None

    def add_api_key(self, key_name, key_value, overwrite=False):
        """
        Adds a new API key to the encrypted file.
        If the key already exists and overwrite is False, it raises an error.
        """
        if self.use_env_only:
            print("Environment keys are in use. Skipping encrypted file modifications.")
            return

        data = self._decrypt_file()
        if key_name in data and not overwrite:
            print(f"Key '{key_name}' already exists. Use overwrite=True to replace it.")
            return

        data[key_name] = key_value
        self._encrypt_file(data)
        print(f"Key '{key_name}' {'updated' if key_name in data else 'added'} successfully.")

