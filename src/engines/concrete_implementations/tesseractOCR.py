import pytesseract
from ..IEngine import OCREngine # Assuming IEngine is in parent directory 'engines'
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Dict, Any
import re # For parsing confidence from image_to_string if needed

logger = logging.getLogger(__name__)

LANG_CODE_MAPPING = {
    "ar": "ara",
    "en": "eng",
}

class TesseractOCREngine(OCREngine):
    def __init__(self, lang_list: List[str], **kwargs):
        super().__init__(lang_list, **kwargs)
        self.tesseract_lang_str = "+".join([LANG_CODE_MAPPING.get(l, l) for l in self.lang_list])
        self.tesseract_config = self.configs.get("tesseract_config", "")
        # Add a specific config for recognizing single lines, e.g., PSM 7 or 8
        self.tesseract_single_line_config = self.configs.get("tesseract_single_line_config", "--psm 7")


        tesseract_cmd = self.configs.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        logger.info(f"TesseractOCR engine initialized for languages: {self.tesseract_lang_str}. Config: '{self.tesseract_config}', Single Line Config: '{self.tesseract_single_line_config}'")

    def _recognize_text_in_region(self, image_crop: Image.Image) -> Dict[str, Any]:
        """
        Recognizes text in a given image crop (assumed to be a single text line).
        Returns:
            {'text': str, 'text_confidence': float}
        """
        # Using image_to_data to get word-level confidences and then averaging for the line
        # PSM 7: Treat the image as a single text line.
        # PSM 8: Treat the image as a single word.
        # PSM 13: Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
        # Using PSM 7 is generally good for lines.
        try:
            data = pytesseract.image_to_data(
                image_crop,
                lang=self.tesseract_lang_str,
                config=self.tesseract_single_line_config, # Use specific config for single lines
                output_type=pytesseract.Output.DICT
            )
            
            words = []
            confidences = []
            for i in range(len(data['level'])):
                # Only consider words (level 5) that have actual text
                if data['level'][i] == 5 and data['text'][i].strip() and int(data['conf'][i]) > -1:
                    words.append(data['text'][i].strip())
                    confidences.append(float(data['conf'][i]))
            
            if not words:
                return {'text': "", 'text_confidence': 0.0}

            recognized_text = " ".join(words)
            # Average confidence of detected words in the line
            line_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {'text': recognized_text, 'text_confidence': line_confidence / 100.0} # Normalize to 0-1

        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract error during region recognition: {e}")
            return {'text': "", 'text_confidence': 0.0}
        except Exception as e:
            logger.error(f"Unexpected error during region recognition: {e}", exc_info=True)
            return {'text': "", 'text_confidence': 0.0}

    def recognize_detected_lines(
        self,
        images: List[Image.Image],
        pages_detected_lines_info: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Performs OCR on pre-detected text lines for multiple images.
        Augments the input detected_lines_info with 'text' and 'text_confidence'.

        Args:
            images: List of PIL Image objects (full pages).
            pages_detected_lines_info: List of pages. Each page is a list of dictionaries,
                                       where each dict has at least 'bbox'.
                                       Example input structure:
                                       [ # Page 1
                                         [ {'bbox': [x1,y1,x2,y2], ...other_keys...}, ... ],
                                         # Page 2
                                         [ {'bbox': [x1,y1,x2,y2], ...other_keys...}, ... ],
                                       ]
        Returns:
            List of pages, with each detected line dictionary augmented:
            [ # Page 1
              [ {'bbox': ..., 'text': 'recognized', 'text_confidence': 0.95, ...}, ... ], ...
            ]
        """
        if len(images) != len(pages_detected_lines_info):
            logger.error("Mismatch between number of images and number of pages in detected_lines_info.")
            raise ValueError("Number of images must match number of pages in detected_lines_info.")

        output_pages_recognized_lines = []

        for page_idx, full_page_image in enumerate(images):
            detected_lines_on_page = pages_detected_lines_info[page_idx]
            recognized_lines_for_page = []

            if not detected_lines_on_page:
                output_pages_recognized_lines.append([])
                continue

            logger.debug(f"TesseractOCR: Recognizing {len(detected_lines_on_page)} detected lines on page {page_idx + 1}.")
            
            pil_page_image_rgb = full_page_image.convert("RGB")

            for line_info in detected_lines_on_page:
                bbox = line_info.get('bbox')
                if not bbox or not (isinstance(bbox, list) and len(bbox) == 4):
                    logger.warning(f"Skipping line due to invalid bbox: {bbox} on page {page_idx + 1}")
                    # Add original info but with error text, or skip entirely
                    augmented_line_info = {
                        **line_info,
                        'text': "[INVALID_BBOX]",
                        'text_confidence': 0.0
                    }
                    recognized_lines_for_page.append(augmented_line_info)
                    continue

                # Ensure coordinates are integers for cropping
                try:
                    # Add a small padding to the crop, Tesseract sometimes struggles with tight crops
                    padding = 2 
                    x1, y1, x2, y2 = bbox
                    crop_x1 = max(0, int(x1) - padding)
                    crop_y1 = max(0, int(y1) - padding)
                    crop_x2 = min(pil_page_image_rgb.width, int(x2) + padding)
                    crop_y2 = min(pil_page_image_rgb.height, int(y2) + padding)

                    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                        logger.warning(f"Skipping line due to invalid crop dimensions for bbox: {bbox} on page {page_idx+1}. Crop: ({crop_x1},{crop_y1},{crop_x2},{crop_y2})")
                        recognition_result = {'text': "[CROP_ERROR]", 'text_confidence': 0.0}
                    else:
                        image_crop = pil_page_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        recognition_result = self._recognize_text_in_region(image_crop)
                
                except Exception as crop_exc:
                    logger.error(f"Error cropping image for bbox {bbox} on page {page_idx+1}: {crop_exc}")
                    recognition_result = {'text': "[CROP_EXCEPTION]", 'text_confidence': 0.0}


                augmented_line_info = {
                    **line_info, # Keep original keys like 'label', 'confidence' (layout conf), 'position'
                    'text': recognition_result['text'],
                    'text_confidence': recognition_result['text_confidence']
                }
                recognized_lines_for_page.append(augmented_line_info)
            
            output_pages_recognized_lines.append(recognized_lines_for_page)

        return output_pages_recognized_lines

    # --- IEngine Interface Methods ---

    def get_structured_output(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Primary method for getting structured output.
        IF THIS ENGINE IS USED AS A RECOGNIZER ONLY (after external detection),
        this method might not be directly called by the main pipeline.
        The `recognize_detected_lines` would be called instead.

        If it *is* called directly (e.g., Tesseract as a standalone full OCR),
        it performs its own detection (implicit in image_to_data) and word extraction.
        This provides WORD-LEVEL output.
        """
        logger.info("TesseractOCREngine.get_structured_output: Performing full-page OCR (detection + word extraction).")
        all_outputs = []
        for i, image in enumerate(images):
            # This existing method gets word-level output if Tesseract does its own detection
            all_outputs.append(self._get_structured_output_single_image(image))
        return all_outputs

    def _get_structured_output_single_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Internal helper for full-page word-level OCR if Tesseract does detection."""
        # ... (This is your existing implementation that gives WORD level output)
        # ... (It should remain as is if you want Tesseract to sometimes do its own detection)
        logger.debug("TesseractOCR: _get_structured_output_single_image (full page word detection).")
        try:
            data = pytesseract.image_to_data(
                image.convert("RGB"),
                lang=self.tesseract_lang_str,
                config=self.tesseract_config, # General config for full page
                output_type=pytesseract.Output.DICT
            )
            n_boxes = len(data['level'])
            results: List[Dict[str, Any]] = []
            for i in range(n_boxes):
                if data['level'][i] != 5: continue # Level 5 is word
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                if not text or conf < 0: continue
                x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
                results.append({'bbox': [x, y, x + w, y + h], 'text': text, 'confidence': conf / 100.0}) # Normalize conf
            return results
        except pytesseract.TesseractNotFoundError: # ... (error handling)
            logger.error("Tesseract is not installed or not found in your PATH.")
            raise
        except Exception as e:
            logger.error(f"TesseractOCR: Error during full-page structured OCR: {e}")
            return [{'bbox':[], 'text': f"[ERROR_TESS_FULL_OCR: {e}]", 'confidence':0.0}]


    def recognize_text(self, images: List[Image.Image]) -> List[str]:
        """
        Performs full-page OCR and returns concatenated text per page.
        This uses Tesseract's internal detection and line/paragraph assembly.
        """
        logger.info("TesseractOCREngine.recognize_text: Performing full-page OCR (concatenated text).")
        all_texts = []
        for i, image in enumerate(images):
            all_texts.append(self._recognize_text_single_image(image))
        return all_texts

    def _recognize_text_single_image(self, image: Image.Image) -> str:
        # ... (Your existing implementation for full-page text string) ...
        logger.debug("TesseractOCR: _recognize_text_single_image (full page string).")
        try:
            text = pytesseract.image_to_string(
                image.convert("RGB"), 
                lang=self.tesseract_lang_str, 
                config=self.tesseract_config # General config for full page
            )
            return text
        except Exception as e: # ... (error handling)
            logger.error(f"TesseractOCR: Error during full-page text recognition: {e}")
            return f"[ERROR_TESS_FULL_TEXT: {e}]"


    # --- Visualization methods ---
    # These typically display based on _get_structured_output_single_image or provided data.
    # No direct changes needed unless their input assumptions change drastically.

    def _draw_on_image(self, image: Image.Image, structured_output: List[Dict[str, Any]], draw_text: bool = False, title_prefix="TesseractOCR"):
        # ... (your existing drawing logic, ensure it handles 'text_confidence' if you want to display it)
        display_image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(display_image)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()

        for item in structured_output:
            if not isinstance(item.get('bbox'), list) or len(item['bbox']) != 4: continue
            x1, y1, x2, y2 = item['bbox']
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            
            text_to_show_on_image = ""
            if draw_text:
                text_to_show_on_image = item.get('text', '')
            
            # Add confidence display (can be layout confidence or text_confidence)
            display_label = item.get('label', '') # Layout label if present
            primary_confidence = item.get('confidence') # Could be layout confidence
            text_rec_confidence = item.get('text_confidence') # Actual text recognition confidence

            info_str = display_label
            if primary_confidence is not None:
                info_str += f" (C:{primary_confidence:.2f})"
            if text_rec_confidence is not None:
                 info_str += f" (TC:{text_rec_confidence:.2f})"
            
            if not draw_text and info_str.strip(): # If not drawing recognized text, draw the label+conf info
                text_to_show_on_image = info_str.strip()
            elif draw_text and info_str.strip(): # If drawing text, append conf info
                text_to_show_on_image += f" ({info_str.strip()})"


            if text_to_show_on_image:
                text_position = (x1, y1 - 15 if y1 - 15 > 0 else y1 + 5)
                try:
                    text_render_bbox = draw.textbbox(text_position, text_to_show_on_image, font=font)
                    draw.rectangle(text_render_bbox, fill="blue")
                    draw.text(text_position, text_to_show_on_image, fill="white", font=font)
                except Exception as e_draw:
                     logger.warning(f"Could not draw text label '{text_to_show_on_image}': {e_draw}")

        display_image.show(title=f"{title_prefix} Output")


    def display_bounding_boxes(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("TesseractOCR: Displaying bounding boxes.")
        # If structured_output is from the new pipeline (List of augmented line_info dicts),
        # this will display those. If None, it falls back to _get_structured_output_single_image (words).
        if structured_output is None:
            logger.info("TesseractOCR: No structured_output provided to display_bounding_boxes, using internal full-page word detection.")
            structured_output = self._get_structured_output_single_image(image) # Word-level
        self._draw_on_image(image, structured_output, draw_text=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("TesseractOCR: Displaying annotated output.")
        if structured_output is None:
            logger.info("TesseractOCR: No structured_output provided to display_annotated_output, using internal full-page word detection.")
            structured_output = self._get_structured_output_single_image(image) # Word-level
        self._draw_on_image(image, structured_output, draw_text=True)