# coding: utf-8
from cryptography.fernet import Fernet


class EncryptionService:
    def __init__(self, key):
        self._fernet = Fernet(key)

    def encrypt(self, data):
        return self._fernet.encrypt(data)

    def decrypt(self, data):
        return self._fernet.decrypt(data)
