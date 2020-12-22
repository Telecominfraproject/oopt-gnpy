#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.rest_example
=======================

GNPy as a rest API example
'''

import logging
from logging.handlers import RotatingFileHandler

import werkzeug
from flask import Flask
from werkzeug.exceptions import InternalServerError

import gnpy.core.exceptions as exceptions
from gnpy.api.exception.exception_handler import bad_request_handler, common_error_handler

_logger = logging.getLogger(__name__)

from gnpy.api import app


@app.route('/api/v1/status', methods=['GET'])
def api_status():
    return {"version": "v1", "status": "ok"}, 200


def _init_logger():
    handler = RotatingFileHandler('api.log', maxBytes=1024 * 1024, backupCount=5, encoding='utf-8')
    ch = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[handler, ch],
                        format="%(asctime)s %(levelname)s %(name)s(%(lineno)s) [%(threadName)s - %(thread)d] - %("
                               "message)s")


def _init_app():
    app.register_error_handler(KeyError, bad_request_handler)
    app.register_error_handler(TypeError, bad_request_handler)
    app.register_error_handler(ValueError, bad_request_handler)
    app.register_error_handler(exceptions.ConfigurationError, bad_request_handler)
    app.register_error_handler(exceptions.DisjunctionError, bad_request_handler)
    app.register_error_handler(exceptions.EquipmentConfigError, bad_request_handler)
    app.register_error_handler(exceptions.NetworkTopologyError, bad_request_handler)
    app.register_error_handler(exceptions.ServiceError, bad_request_handler)
    app.register_error_handler(exceptions.SpectrumError, bad_request_handler)
    app.register_error_handler(exceptions.ParametersError, bad_request_handler)
    app.register_error_handler(AssertionError, bad_request_handler)
    app.register_error_handler(InternalServerError, common_error_handler)
    for error_code in werkzeug.exceptions.default_exceptions:
        app.register_error_handler(error_code, common_error_handler)


def main():
    _init_logger()
    _init_app()
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
