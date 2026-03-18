MODELS_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODELS_REGISTRY[name] = cls
        return cls
    return decorator