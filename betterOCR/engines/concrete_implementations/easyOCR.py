import easyocr
from engines.IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EasyOCREngine(OCREngine):
    def __init__(self, lang: List[str], **kwargs):
        super().__init__(lang, **kwargs)
        try:
            self.reader = easyocr.Reader(self.lang, **self.configs)
            logger.info(f"EasyOCR engine initialized for languages: {self.lang}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR engine: {e}")
            raise

    def _raw_ocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Helper to get raw OCR output from EasyOCR."""
        
        img_np = np.array(image.convert("RGB")) # Convert to numpy RGB for consistency
        
        # The format is (bbox, text, confidence)
        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        results = self.reader.readtext(img_np)
        
        structured_results = []
        for (bbox_coords, text, conf) in results:

            x_coords = [p[0] for p in bbox_coords]
            y_coords = [p[1] for p in bbox_coords]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            structured_results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'text': text,
                'confidence': conf
            })
        return structured_results

    def recognize_text(self, image: Image.Image) -> str:
        logger.debug("EasyOCR: Starting text recognition.")
        structured_results = self._raw_ocr(image)
        full_text = " ".join([res['text'] for res in structured_results])
        logger.debug(f"EasyOCR: Recognized text: {full_text[:200]}...")
        return full_text

    def get_structured_output(self, image: Image.Image) -> List[Dict[str, Any]]:
        logger.debug("EasyOCR: Getting structured output.")
        return self._raw_ocr(image)

    def _draw_on_image(self, image: Image.Image, structured_output: List[Dict[str, Any]], draw_text: bool = False):
        display_image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(display_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for item in structured_output:
            x1, y1, x2, y2 = item['bbox']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            if draw_text:
                text_to_draw = item.get('text', '')
                text_position = (x1, y1 - 15 if y1 - 15 > 0 else y1 + 5)
                if text_to_draw:
                    bbox = draw.textbbox(text_position, text_to_draw, font=font)
                    draw.rectangle(bbox, fill="red")
                    draw.text(text_position, text_to_draw, fill="white", font=font)
        
        display_image.show(title="EasyOCR Output")


    def display_bounding_boxes(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("EasyOCR: Displaying bounding boxes.")
        if structured_output is None:
            structured_output = self.get_structured_output(image)
        self._draw_on_image(image, structured_output, draw_text=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("EasyOCR: Displaying annotated output.")
        if structured_output is None:
            structured_output = self.get_structured_output(image)
        self._draw_on_image(image, structured_output, draw_text=True)