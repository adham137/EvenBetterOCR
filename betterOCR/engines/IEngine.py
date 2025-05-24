from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class OCREngine(ABC):
    @abstractmethod
    def __init__(self, lang: List[str], **kwargs):
        """
        Initialize the OCR engine.
        Args:
            lang: List of language codes (e.g., ['en', 'ar']).
            **kwargs: Additional engine-specific configurations.
        """
        self.lang = lang
        self.configs = kwargs

    @abstractmethod
    def recognize_text(self, image: Image.Image) -> str:
        """
        Perform OCR on the given image and return the recognized text as a single string.
        Args:
            image: A PIL Image object to process.
        Returns:
            A string containing all recognized text from the image.
        """
        pass

    @abstractmethod
    def get_structured_output(self, image: Image.Image) -> List[dict]:
        """
        Perform OCR and return a structured output.
        Each dict should contain 'bbox' (e.g., [x1, y1, x2, y2]) and 'text'.
        Args:
            image: A PIL Image object to process.
        Returns:
            A list of dictionaries, where each dictionary represents a detected text block
            and contains its bounding box and recognized text.
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



