# coding: utf-8


class Error:

    def __init__(self, code: int = None, message: str = None, description: str = None):
        """Error
        :param code: The code of this Error.
        :type code: int
        :param message: The message of this Error.
        :type message: str
        :param description: The description of this Error.
        :type description: str
        """
        self._code = code
        self._message = message
        self._description = description
