from cryptography.fernet import Fernet


# Generate a key for encryption, store it securely, and reuse for encryption/decryption
def generate_key():
    return Fernet.generate_key()

def encrypt_file(file_path, encryption_key):
    with open(file_path, 'rb') as file:
        data = file.read()
    fernet = Fernet(encryption_key)
    encrypted_data = fernet.encrypt(data)
    with open(file_path + '.enc', 'wb') as file:
        file.write(encrypted_data)

