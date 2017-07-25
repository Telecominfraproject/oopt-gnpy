class Params:
    def __init__(self, *args):
        req_params = args[0]
        params = args[1].get('parameters')
        missing_params = list(set(req_params) - set(params.keys()))
        if len(missing_params):
            print("missing params:", ','.join(missing_params))
            raise ValueError
        for k, v in params.items():
            setattr(self, k, v)


class NetworkElement:

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.type = kwargs.get('type')
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        self.params = Params(self.required_params, kwargs)


class Fiber(NetworkElement):
    required_params = ['length', 'loss']


class Edfa_boost(NetworkElement):
    required_params = ['gain', 'nf']


class Edfa_line(NetworkElement):
    required_params = ['gain', 'nf']


class Edfa_preamp(NetworkElement):
    required_params = ['gain', 'nf']


class Tx(NetworkElement):
    required_params = ['channels']
