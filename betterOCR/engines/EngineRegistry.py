# betterOCR/engines/EngineRegistry.py
import logging
logger = logging.getLogger(__name__)

class EngineRegistry:
    def __init__(self):
        self.engines = {} # Stores name: engine_class
        logger.debug("EngineRegistry initialized.")
        
    def register_engine(self, name: str, engine_class: type):
        if not callable(engine_class):
            logger.error(f"Attempted to register non-class {engine_class} for engine {name}.")
            raise TypeError("engine_class must be a class type, not an instance.")
        self.engines[name] = engine_class
        logger.info(f"Engine '{name}' registered with class {engine_class.__name__}.")
        
    def get_engine_class(self, name: str) -> type:
        engine_class = self.engines.get(name)
        if not engine_class:
            logger.error(f"Engine class for '{name}' not found in registry.")
            raise ValueError(f"Engine class {name} not found")
        logger.debug(f"Retrieved engine class '{name}'.")
        return engine_class

    def get_engine_instance(self, name: str, *args, **kwargs): 
        engine_class = self.get_engine_class(name)
        logger.debug(f"Instantiating engine '{name}'...")
        try:
            instance = engine_class(*args, **kwargs)
            logger.info(f"Engine '{name}' instantiated successfully.")
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate engine '{name}': {e}")
            raise