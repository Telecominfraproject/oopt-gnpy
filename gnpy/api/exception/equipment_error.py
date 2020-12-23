# coding: utf-8


class EquipmentError(Exception):
    """ Exception raise for equipment error
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message