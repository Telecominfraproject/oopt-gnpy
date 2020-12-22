from flask import Flask

app = Flask(__name__)

import gnpy.api.route.path_request_route
