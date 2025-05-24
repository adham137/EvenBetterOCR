class EngineRegistry:
    def __init__(self):
        self.engines = {}
        
    def register_engine(self, name, engine_class):
        self.engines[name] = engine_class
        
    def get_engine(self, name, *args, **kwargs):
        engine_class = self.engines.get(name)
        if not engine_class:
            raise ValueError(f"Engine {name} not found")
        return engine_class(*args, **kwargs)