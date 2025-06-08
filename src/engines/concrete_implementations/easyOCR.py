import easyocr
from ..IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EasyOCREngine(OCREngine):
    def __init__(self, lang_list: List[str], **kwargs):
        super().__init__(lang_list, **kwargs)
        try:
            self.reader = easyocr.Reader(self.lang_list, **self.configs)
            # logger.info(f"EasyOCR engine initialized for languages: {self.lang}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR engine: {e}")
            raise

    def _raw_ocr(self, images: Image.Image) -> List[Dict[str, Any]]:
        """
        Helper to get raw OCR output from EasyOCR.

        Returns:
            A list of dicts, each with:
            - 'bbox': [x1, y1, x2, y2]
            - 'text': the recognized string
            - 'confidence': float confidence
        """
        img_np = np.array(image.convert("RGB"))
        results = self.reader.readtext(img_np)  # [(bbox, text, conf), ...]
        
        structured_results = []
        for bbox_coords, text, conf in results:
            xs = [p[0] for p in bbox_coords]
            ys = [p[1] for p in bbox_coords]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            structured_results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'text': text,
                'confidence': conf
            })
        return structured_results

 
    def recognize_text(self, images: List[Image.Image]) -> List[str]:
        """
        Runs EasyOCR, then re-assembles its flat word list into lines by
        clustering on vertical position.  Inserts '\n' between inferred lines.

        Algorithm:
            1. Pull raw words with bboxes.
            2. Sort all words by their y-coordinate (top), then x-coordinate (left).
            3. Compute a “line threshold” (≈ median word-height).
            4. Walk the sorted words, grouping into the same line if their
                top‐y is within threshold of the current line’s baseline.
            5. Within each line, sort by x and join with spaces.
            6. Finally, join lines with '\n'.

        Returns:
            A List of single strings containing recognized text, with '\n' between lines.
        """
        logger.debug("EasyOCR: Starting text recognition.")
        res = []
        for image in images:
          words = self._raw_ocr(image)

          if not words:
              return ""

          # 1) Sort tokens top→bottom, left→right
          words.sort(key=lambda w: (w['bbox'][1], w['bbox'][0]))

          # 2) Estimate a threshold for “same line” as a fraction of median height
          heights = [b[3] - b[1] for b in (w['bbox'] for w in words)]
          median_h = float(np.median(heights))
          line_thresh = max(1.0, median_h * 0.8)

          # 3) Group into lines
          lines: List[List[Dict[str, Any]]] = []
          current_line = [words[0]]
          baseline = words[0]['bbox'][1]

          for w in words[1:]:
              top_y = w['bbox'][1]
              if abs(top_y - baseline) <= line_thresh:
                  current_line.append(w)
                  # optionally update baseline to average or keep first word’s top
              else:
                  lines.append(current_line)
                  current_line = [w]
                  baseline = top_y
          lines.append(current_line)

          # 4) Build the final text
          line_strs: List[str] = []
          for line in lines:
              # sort each line by x-coordinate
              line.sort(key=lambda w: w['bbox'][0])
              texts = [w['text'] for w in line]
              line_strs.append(" ".join(texts))

          full_text = "\n".join(line_strs)
          logger.debug(f"EasyOCR: Recognized text (with newlines):\n{full_text[:200]}…")
          res.append(full_text)
        return res
    def get_structured_output(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        logger.debug("EasyOCR: Getting structured output.")
        return [self._raw_ocr(image) for image in images]

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