# coding: utf-8
from flask import Flask

app = Flask(__name__)

import gnpy.api.route.path_request_route
import gnpy.api.route.status_route
import gnpy.api.route.topology_route
import gnpy.api.route.equipments_route
