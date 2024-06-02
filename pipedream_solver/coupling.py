class SpeciesCoupling():
    def __init__(self, species={}, functions=[], parameters={}):
        self.species = {}
        self.functions = []
        self.parameters = {}
        for key, value in species.items():
            if hasattr(self, key):
                raise NameError
            setattr(self, key, value)
            self.species[key] = value
        for function in functions:
            if hasattr(self, function.__name__):
                raise NameError
            setattr(self, function.__name__, function)
            self.functions.append(function)
        for key, value in parameters.items():
            if hasattr(self, key):
                raise NameError
            setattr(self, key, value)
            self.parameters[key] = value

    def step(self, **kwargs):
        '''
        Steps through all functions in `self.functions` in order.
        '''
        for function in self.functions:
            arg_keys = function.__code__.co_varnames
            # NOTE: Has potential to fail silently
            arg_values = (getattr(self, key, None) for key in arg_keys)
            function_inputs = dict(zip(arg_keys, arg_values))
            function_inputs.update(kwargs)
            result = function(**function_inputs)
