from typing import List, Dict, Any
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor # For more detailed layout if needed later
import torch
import os
from engines.IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import logging
import numpy as np

# os.environ['COMPILE_LAYOUT']='true'
# os.environ['LAYOUT_BATCH_SIZE']='16'

logger = logging.getLogger(__name__)

class SuryaOCREngine(OCREngine):
    def __init__(self, lang: List[str], **kwargs):
        super().__init__(lang, **kwargs)                # lang not important to surya currently
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SuryaOCR: Using device: {self.device}")
    
        try:
            self.recognition_predictor = RecognitionPredictor()
            self.detection_predictor = DetectionPredictor()
            self.detection_predictor.model = self.detection_predictor.model.to(device=self.device, dtype=torch.float32)
            logger.info("SuryaOCR engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SuryaOCR engine: {e}")
            raise

        
    def _get_text_detections(self, image: Image.Image) -> List[Any]: 
        predictions =   self.recognition_predictor([image.convert("RGB")], det_predictor=self.detection_predictor, return_words=True)[0]   # Ensure RGB and return words
        if predictions and hasattr(predictions, 'text_lines'):
            return predictions                 # assuming single image processing
        return []
    
    def _get_line_detections(self, image: Image.Image) -> List[Any]: 
        predictions =   self.detection_predictor([image.convert("RGB")])[0]   # Ensure RGB
        if predictions and hasattr(predictions, 'bboxes'):
            return predictions.bboxes                 # assuming single image processing
        return []

    def get_structured_output(self, image: Image.Image) -> List[Dict[str, Any]]:
        logger.debug("SuryaOCR: Getting structured output.")
        img_rgb = image.convert("RGB")
        detections = self._get_text_detections(img_rgb)
        
        words_dict = []
        for line in detections.text_lines:
            if line.words:
                for word in line.words:
                    if word.bbox_valid:
                        words_dict.append({
                            'bbox': word.bbox,
                            'text': word.text,
                            'confidence': word.confidence
                        })
        logger.debug(f"SuryaOCR: Structured output generated with {len(words_dict)} items.")
        return words_dict
        
    def recognize_text(self, image: Image.Image) -> str:
        logger.debug("SuryaOCR: Starting text recognition.")
        lines = self._get_text_detections(self, image).text_lines
        full_text = "\n".join([line.text for line in lines])  # you can add a confidence threshold here also
        logger.debug(f"SuryaOCR: Recognized text: {full_text[:200]}...")
        return full_text

    def _draw_on_image(self, image: Image.Image, items_to_draw: List[Dict[str, Any]], draw_text_content: bool = False, box_color="lime", text_color="black", font_size=10):
        img_display = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img_display)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default() 

        for item in items_to_draw:
            x1, y1, x2, y2 = map(int, item['bbox'])
            current_box_color = box_color
            
            draw.rectangle([x1, y1, x2, y2], outline=current_box_color, width=2) # [cite: 54]
            if draw_text_content:
                text = item.get('text', "")
                if not text and 'confidence' in item: # Fallback to confidence if no text for annotation
                    text = f"Conf: {item['confidence']:.2f}"
                
                if text:
                    text_bbox = draw.textbbox((x1, y1), text, font=font)
                    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    draw.rectangle(
                        [x1, y1 - text_h - 4, x1 + text_w + 4, y1], 
                        fill=current_box_color
                    )
                    draw.text((x1 + 2, y1 - text_h - 2), text, fill=text_color, font=font)
        
        img_display.show(title="SuryaOCR Output")


    def display_textline_boxes(self, image: Image.Image):
        logger.info("SuryaOCR: Displaying text lines bounding boxes.")
        detections = self._get_line_detections(image)
        items_to_draw = [{'bbox': det.bbox, 'confidence': det.confidence} for det in detections]
        self._draw_on_image(image, items_to_draw, draw_text_content=False)

    def display_bounding_boxes(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("SuryaOCR: Displaying bounding boxes.")
        if structured_output is None:
            detections = self.get_structured_output(image)
            items_to_draw = [{'bbox': det.bbox, 'confidence': det.confidence} for det in detections]
        else:
            items_to_draw = structured_output
        self._draw_on_image(image, items_to_draw, draw_text_content=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("SuryaOCR: Displaying annotated output.")
        if structured_output is None:
            structured_output = self.get_structured_output(image)
        self._draw_on_image(image, structured_output, draw_text_content=True)
