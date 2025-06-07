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

# os.environ['COMPILE_DETECTOR']='true'
# os.environ['COMPILE_RECOGNITION']='true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DETECTOR_BATCH_SIZE = 3     # Each batch takes 440mb of vram, calculate your own (Keep in mind that the models themselves take 2GB of vram)
RECOGNITION_BATCH_SIZE = 100 # Each batch takes 40mb of vram, calculate your own (Keep in mind that the models themselves take 2GB of vram)
# os.environ['LAYOUT_BATCH_SIZE']='16'

logger = logging.getLogger(__name__)

class SuryaOCREngine(OCREngine):
    def __init__(self, lang_list: List[str], **kwargs):
        super().__init__(lang_list, **kwargs)                # lang not important to surya currently
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SuryaOCR: Using device: {self.device}")

        self.page_processing_batch_size = kwargs.get("page_processing_batch_size", None) # Default to None if not provided
        if self.page_processing_batch_size is not None:
            logger.info(f"SuryaOCR: Will process images in batches of {self.page_processing_batch_size} pages.")
        try:
            self.recognition_predictor = RecognitionPredictor()
            self.detection_predictor = DetectionPredictor(dtype=torch.float32)
            # self.detection_predictor.model = self.detection_predictor.model.to(device=self.device, dtype=torch.float32)
            logger.info("SuryaOCR engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SuryaOCR engine: {e}")
            raise

        
    def _get_text_detections(self, images: List[Image.Image]) -> List[List[Any]]: 
        with torch.no_grad(): 
            predictions =   self.recognition_predictor([image.convert("RGB") for image in images], det_predictor=self.detection_predictor, return_words=True, detection_batch_size= DETECTOR_BATCH_SIZE, recognition_batch_size= RECOGNITION_BATCH_SIZE)   # Ensure RGB and return words
        if len(predictions) and hasattr(predictions[0], 'text_lines'):
            return predictions                 # return a list of predictions (one for each page)
        return []
    
    def _get_line_detections(self, images: List[Image.Image]) -> List[List[Any]]: 
        predictions =   self.detection_predictor([image.convert("RGB") for image in images]) # Ensure RGB
        if len(predictions) and hasattr(predictions[0], 'bboxes'):
            return predictions                 # return a list of predictions (one for each page)
        return []

    def get_structured_output(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        logger.debug("SuryaOCR: Getting structured output.")
        detections = self._get_text_detections(images)
        
        pages = []
        # words_dict = []
        for page in detections:
            words_dict = []
            for line in page.text_lines:
                if line.words:
                    for word in line.words:
                        if word.bbox_valid:
                            words_dict.append({
                                'bbox': word.bbox,
                                'text': word.text,
                                'confidence': word.confidence
                            })
            pages.append(words_dict)
        logger.debug(f"SuryaOCR: Structured output generated with {len(pages)} pages.")
        return pages
        
    # def recognize_text(self, images: List[Image.Image]) -> List[str]:
    #     logger.debug("SuryaOCR: Starting text recognition.")
    #     detections = self._get_text_detections(images)
    #     pages = []
    #     for page in detections:
    #         lines = page.text_lines
    #         full_page_text = "\n".join([line.text for line in lines])  # you can add a confidence threshold here also
    #         pages.append(full_page_text)
    #     logger.debug(f"SuryaOCR: Recognized text for {len(detections)} pages")
    #     return pages


    def recognize_text(self, images: List[Image.Image]) -> List[str]:
        logger.debug(f"SuryaOCR: Starting text recognition for {len(images)} images.")
        if not images:
            return []

        all_page_texts: List[str] = []

        if self.page_processing_batch_size is None or self.page_processing_batch_size >= len(images):
            # Process all at once (old behavior or small number of images)
            detections = self._get_text_detections(images)
            for page_detection in detections:
                lines = page_detection.text_lines
                full_page_text = "\n".join([line.text for line in lines])
                all_page_texts.append(full_page_text)
        else:
            # Process in batches
            num_images = len(images)
            for i in range(0, num_images, self.page_processing_batch_size):
                chunk_images = images[i : i + self.page_processing_batch_size]
                logger.debug(f"SuryaOCR: Processing page chunk {i//self.page_processing_batch_size + 1}/{(num_images + self.page_processing_batch_size - 1)//self.page_processing_batch_size} (images {i+1}-{min(i+self.page_processing_batch_size, num_images)})")
                
                # print(f" Before detecting text, Batch ({i}), Memory({torch.cuda.memory_summary()})")
                chunk_detections_gpu = self._get_text_detections(chunk_images)
                # print(f" After detecting text, Batch ({i}), Memory({torch.cuda.memory_summary()})")
                chunk_detections = chunk_detections_gpu.detach().cpu() if isinstance(chunk_detections_gpu, torch.Tensor) else chunk_detections_gpu
                
                for page_detection in chunk_detections:
                    lines = page_detection.text_lines
                    full_page_text = "\n".join([line.text for line in lines])
                    all_page_texts.append(full_page_text)
                
                # Clean up memory after processing a chunk
                del chunk_images
                del chunk_detections_gpu
                del chunk_detections
                import gc
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                # print(f" After emptyinh cache, Batch ({i}), Memory({torch.cuda.memory_summary()})")
                logger.debug(f"SuryaOCR: Cleaned memory after processing chunk.")

        logger.debug(f"SuryaOCR: Recognized text for {len(all_page_texts)} pages.")
        return all_page_texts

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
            
            draw.rectangle([x1, y1, x2, y2], outline=current_box_color, width=2)
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
        detections = self._get_line_detections([image])
        items_to_draw = [{'bbox': det.bbox, 'confidence': det.confidence} for det in detections[0].bboxes]
        self._draw_on_image(image, items_to_draw, draw_text_content=False)

    def display_bounding_boxes(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        # if the structured output is provided, it must be of only one image
        logger.info("SuryaOCR: Displaying bounding boxes.")
        if structured_output is None:
            detections = self.get_structured_output([image])[0]
            items_to_draw = [{'bbox': det.bbox, 'confidence': det.confidence} for det in detections]
        else:
            items_to_draw = structured_output
        self._draw_on_image(image, items_to_draw, draw_text_content=False)

    def display_annotated_output(self, image: Image.Image, structured_output: List[Dict[str, Any]] = None):
        logger.info("SuryaOCR: Displaying annotated output.")
        if structured_output is None:
            structured_output = self.get_structured_output([image])[0]
        self._draw_on_image(image, structured_output, draw_text_content=True)
