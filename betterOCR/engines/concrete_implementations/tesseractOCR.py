import pytesseract
from engines.IEngine import OCREngine
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
    def __init__(self, lang: List[str], **kwargs):
        # Tesseract languages are typically specified as a single string, e.g., 'eng+ara'
        super().__init__("+".join([LANG_CODE_MAPPING[l] for l in lang if l in LANG_CODE_MAPPING]), **kwargs) ### Special language mapping for tesseract
        self.tesseract_config = self.configs.get("tesseract_config", "") 
        
        tesseract_cmd = self.configs.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        logger.info(f"TesseractOCR engine initialized for languages: {self.lang}. Config: '{self.tesseract_config}'")

    def get_structured_output(self, image: Image.Image) -> List[Dict[str, Any]]:
        logger.debug("TesseractOCR: Getting structured output.")
        try:
            data_str = pytesseract.image_to_data(
                image.convert("RGB"), 
                lang=self.lang, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.STRING
            )
            
            # Parse the TSV data
            from io import StringIO
            df = pd.read_csv(StringIO(data_str), sep='\t')
            
            results = []
            # Filter out entries with low confidence or no text
            df_filtered = df[df.conf > -1] 

            for index, row in df_filtered.iterrows():
                if pd.notna(row['text']) and str(row['text']).strip():
                    x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                    results.append({
                        'bbox': [x, y, x + w, y + h],
                        'text': str(row['text']),
                        'confidence': float(row['conf'])
                        # Tesseract data also has 'level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num'
                    })
            logger.debug(f"TesseractOCR: Structured output generated with {len(results)} items.")
            return results
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not found in your PATH.")
            raise
        except Exception as e:
            logger.error(f"TesseractOCR: Error during OCR processing: {e}")
            return [] 

    def recognize_text(self, image: Image.Image) -> str:
        logger.debug("TesseractOCR: Starting text recognition.")
        try:
            text = pytesseract.image_to_string(
                image.convert("RGB"), 
                lang=self.lang, 
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
            structured_output = self.get_structured_output(image)
        self._draw_on_image(image, structured_output, draw_text=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("TesseractOCR: Displaying annotated output.")
        if structured_output is None:
            structured_output = self.get_structured_output(image)
        self._draw_on_image(image, structured_output, draw_text=True)