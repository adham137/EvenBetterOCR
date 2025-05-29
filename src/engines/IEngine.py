from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image

class OCREngine(ABC):
    @abstractmethod
    def __init__(self, lang_list: List[str], **kwargs):
        """
        Initialize the OCR engine.
        Args:
            langs: List of language codes (e.g., ['en', 'ar']).
            **kwargs: Additional engine-specific configurations.
        """
        self.lang_list = lang_list
        self.configs = kwargs

    @abstractmethod
    def recognize_text(self, images: List[Image.Image]) -> List[str]:
        """
        Perform OCR on the given list of images and return recognized text for each.
        Args:
            images: A list of PIL Image objects to process.
        Returns:
            A list of strings, where each string contains all recognized text from the corresponding image.
        """
        pass

    @abstractmethod
    def get_structured_output(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Perform OCR and return a structured output for each image.
        Each dict should contain 'bbox' (e.g., [x1, y1, x2, y2]) and 'text'.
        Args:
            images: A list of PIL Image objects to process.
        Returns:
            A list of lists of dictionaries. Outer list corresponds to images.
            Inner list contains detected text blocks for that image.
        """
        pass

    @abstractmethod
    def display_bounding_boxes(self, image: Image.Image, structured_output: List[dict] = None):
        """
        Displays the image with bounding boxes for detected text regions.
        Boxes can be from a provided structured_output or by running detection.
        Args:
            image: A PIL Image object to display.
            structured_output: Optional pre-computed structured output.
                               If None, the engine should run detection itself.
        """
        pass

    @abstractmethod
    def display_annotated_output(self, image: Image.Image, structured_output: List[dict] = None):
        """
        Displays the image with bounding boxes and the recognized text printed near them.
        Args:
            image: A PIL Image object to display.
            structured_output: Optional pre-computed structured output with text and boxes.
                               If None, the engine should run OCR itself.
        """
        pass



