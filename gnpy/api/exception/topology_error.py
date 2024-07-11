# coding: utf-8


class TopologyError(Exception):
    """ Exception raise for topology error
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message