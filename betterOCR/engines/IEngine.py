from abc import ABC, abstractmethod

class OCREngine(ABC):
    @abstractmethod
    def __init__(self, lang):
        pass
    
    @abstractmethod
    def execute(self, image_path):
        pass
    
    # @abstractmethod
    # def detect_boxes(self, image_path):
    #     pass



