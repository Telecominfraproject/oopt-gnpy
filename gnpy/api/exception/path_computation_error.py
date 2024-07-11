# coding: utf-8


class PathComputationError(Exception):
    """ Exception raise for path computation error error
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message