import pytesseract
from ..IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Dict, Any
import pandas as pd # For parsing Tesseract's TSV output

logger = logging.getLogger(__name__)


LANG_CODE_MAPPING = {
    "ar": "ara",
    "en": "eng",
}

class TesseractOCREngine(OCREngine):
    def __init__(self, lang_list: List[str], **kwargs): # Takes lang_list
        super().__init__(lang_list, **kwargs) # Passes lang_list to super
        
        # Create Tesseract-specific language string (e.g., 'eng+ara')
        self.tesseract_lang_str = "+".join([LANG_CODE_MAPPING.get(l, l) for l in self.lang_list])
        
        self.tesseract_config = self.configs.get("tesseract_config", "") 
        
        tesseract_cmd = self.configs.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        logger.info(f"TesseractOCR engine initialized for languages: {self.tesseract_lang_str}. Config: '{self.tesseract_config}'")

    def get_structured_output(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        logger.debug(f"TesseractOCR: Getting structured output for {len(images)} images.")
        all_outputs = []
        for i, image in enumerate(images):
            logger.debug(f"TesseractOCR: Processing image {i+1}/{len(images)} for structured output.")
            all_outputs.append(self._get_structured_output_single_image(image))
        return all_outputs

    def recognize_text(self, images: List[Image.Image]) -> List[str]:
        logger.debug(f"TesseractOCR: Starting text recognition for {len(images)} images.")
        all_texts = []
        for i, image in enumerate(images):
            logger.debug(f"TesseractOCR: Processing image {i+1}/{len(images)} for text.")
            all_texts.append(self._recognize_text_single_image(image))
        return all_texts

    def _get_structured_output_single_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        logger.debug("TesseractOCR: Getting structured output.")
        try:
            # 1) Ask Tesseract for a dict of lists instead of TSV
            data = pytesseract.image_to_data(
                image.convert("RGB"),
                lang=self.tesseract_lang_str,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            n_boxes = len(data['level'])
            results: List[Dict[str, Any]] = []
            seen = set()

            # 2) Loop through all boxes, but only take word (level=5)
            for i in range(n_boxes):
                if data['level'][i] != 5:
                    continue
                
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                if not text or conf < 0:
                    continue

                x, y = int(data['left'][i]), int(data['top'][i])
                w, h = int(data['width'][i]), int(data['height'][i])
                bbox = (x, y, x + w, y + h)

                # 3) Dedupe exact repeats
                key = (text, bbox)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    'bbox': [*bbox],
                    'text': text,
                    'confidence': conf
                })

            logger.debug(f"TesseractOCR: Structured output generated with {len(results)} items.")
            return results

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not found in your PATH.")
            raise
        except Exception as e:
            logger.error(f"TesseractOCR: Error during OCR processing: {e}")
            return []

    def _recognize_text_single_image(self, image: Image.Image) -> str:
        logger.debug("TesseractOCR: Starting text recognition.")
        try:
            text = pytesseract.image_to_string(
                image.convert("RGB"), 
                lang=self.tesseract_lang_str, 
                config=self.tesseract_config
            )
            logger.debug(f"TesseractOCR: Recognized text: {text[:200]}...")
            return text
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not found in your PATH.")
            raise
        except Exception as e:
            logger.error(f"TesseractOCR: Error during text recognition: {e}")
            return "" 

    def _draw_on_image(self, image: Image.Image, structured_output: List[Dict[str, Any]], draw_text: bool = False):
        display_image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(display_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()


        for item in structured_output:
            x1, y1, x2, y2 = item['bbox']
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2) # Blue for Tesseract
            if draw_text:
                text_to_draw = item.get('text', '')
                if text_to_draw:
                    text_position = (x1, y1 - 15 if y1 - 15 > 0 else y1 + 5)
                    bbox = draw.textbbox(text_position, text_to_draw, font=font)
                    draw.rectangle(bbox, fill="blue")
                    draw.text(text_position, text_to_draw, fill="white", font=font)
        display_image.show(title="TesseractOCR Output")

    def display_bounding_boxes(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("TesseractOCR: Displaying bounding boxes.")
        if structured_output is None:
            structured_output = self._get_structured_output_single_image(image)
        self._draw_on_image(image, structured_output, draw_text=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("TesseractOCR: Displaying annotated output.")
        if structured_output is None:
            structured_output = self._get_structured_output_single_image(image)
        self._draw_on_image(image, structured_output, draw_text=True)