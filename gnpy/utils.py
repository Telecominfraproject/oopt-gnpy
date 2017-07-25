import json


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def read_config(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print("File not found:", filepath)
