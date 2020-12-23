# coding: utf-8
import json
import re

import werkzeug

from gnpy.api.model.error import Error

_reaesc = re.compile(r'\x1b[^m]*m')


def common_error_handler(exception):
    """

    :type exception: Exception

    """
    status_code = 500
    if not isinstance(exception, werkzeug.exceptions.HTTPException):
        exception = werkzeug.exceptions.InternalServerError()
        exception.description = "Something went wrong on our side."
    else:
        status_code = exception.code
    response = Error(message=exception.name, description=exception.description,
                     code=status_code)

    return werkzeug.Response(response=json.dumps(response.__dict__), status=status_code, mimetype='application/json')


def bad_request_handler(exception):
    response = Error(message='bad request', description=_reaesc.sub('', str(exception)),
                     code=400)
    return werkzeug.Response(response=json.dumps(response.__dict__), status=400, mimetype='application/json')
