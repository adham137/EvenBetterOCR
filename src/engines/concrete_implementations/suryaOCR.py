from typing import List, Dict, Any
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor # For more detailed layout if needed later
import torch
import os
from ..IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import logging
import numpy as np

# os.environ['COMPILE_DETECTOR']='true'
# os.environ['COMPILE_RECOGNITION']='true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# (Keep in mind that the detection + recognition models themselves take 2GB of vram)
# (Keep in mind that the detection + layout models themselves take 0.7GB of vram)
DETECTOR_BATCH_SIZE = 4      # Each batch takes 440mb of vram, calculate your own 
RECOGNITION_BATCH_SIZE = 15 # Each batch takes 40mb of vram, calculate your own (Keep in mind that the models themselves take 2GB of vram)
LAYOUT_BATCH_SIZE = 8       # Each batch takes 220mb of vram, calculate your own (Keep in mind that the models themselves take 2GB of vram)
# os.environ['LAYOUT_BATCH_SIZE']='16'

logger = logging.getLogger(__name__)

class SuryaOCREngine(OCREngine):
    def __init__(self, lang_list: List[str], use_recognizer: bool = False, **kwargs): # Added use_recognizer
        super().__init__(lang_list, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SuryaOCR: Using device: {self.device}")

        self.page_processing_batch_size = kwargs.get("page_processing_batch_size", None)
        if self.page_processing_batch_size is not None:
            logger.info(f"SuryaOCR: Will process images in batches of {self.page_processing_batch_size} pages.")

        self.use_recognizer = use_recognizer
        self.recognition_predictor = None
        self.layout_predictor = None 
        self.detection_predictor = None 

        
    def _get_raw_text_line_detections(self, images: List[Image.Image]) -> List[Any]: 
        """Internal helper to get raw text line detections for a batch of images."""
        if self.detection_predictor is None:
            logger.error("Surya Detection Predictor not initialized.")
            raise RuntimeError("Surya Detection Predictor not initialized.")
        pil_images = [img.convert("RGB") for img in images]
        with torch.no_grad():
            predictions = self.detection_predictor(pil_images, batch_size=DETECTOR_BATCH_SIZE) 
        return predictions
    
    def _get_line_detections(self, images: List[Image.Image]) -> List[List[Any]]: 
        predictions =   self.detection_predictor([image.convert("RGB") for image in images], batch_size= DETECTOR_BATCH_SIZE) # Ensure RGB
        if len(predictions) and hasattr(predictions[0], 'bboxes'):
            return predictions                 # return a list of predictions (one for each page)
        return []
    
    def _get_layout_predictions(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Performs layout detection on a list of images and returns a structured output.
        Args:
            images: A list of PIL Image objects.
        Returns:
            A list of lists of dictionaries. Outer list corresponds to images.
            Inner list contains detected layout blocks for that image, each with:
            {'bbox': [x1, y1, x2, y2], 'label': str, 'confidence': float, 'position': int}
        """
        if self.layout_predictor is None:
            # This should ideally not happen if constructor ensures it or raises
            logger.error("Surya Layout Predictor is not initialized.")
            raise RuntimeError("Surya Layout Predictor is not initialized.")

        if not images:
            return []

        pil_images = [img.convert("RGB") for img in images]
        
        all_pages_layout_data = []
        batch_size = self.page_processing_batch_size or len(pil_images) 

        for i in range(0, len(pil_images), batch_size):
            chunk_images = pil_images[i : i + batch_size]
            logger.debug(f"SuryaLayout: Processing layout for image chunk {i//batch_size + 1}/{(len(pil_images) + batch_size - 1)//batch_size}")
            
            with torch.no_grad():
                layout_results_batch = self.layout_predictor(chunk_images, batch_size=LAYOUT_BATCH_SIZE) # Returns List[LayoutResult]

            for layout_result in layout_results_batch: # layout_result is one LayoutResult object
                page_layout_data = []
                if layout_result and layout_result.bboxes: # bboxes is a list of LayoutBox
                    for layout_box in layout_result.bboxes:
                        # Ensure bbox is List[float] or List[int]
                        bbox_coords = [float(c) for c in layout_box.bbox]
                        page_layout_data.append({
                            'bbox': [int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])],
                            'label': layout_box.label,
                            'confidence': float(layout_box.confidence),
                            'position': int(layout_box.position) # Reading order
                        })
                all_pages_layout_data.append(page_layout_data)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return all_pages_layout_data

    def get_structured_output(self, images: List[Image.Image], input_detections: List[List[Dict[str, Any]]] ) -> List[List[Dict[str, Any]]]:
        logger.debug("SuryaOCR: Getting structured output.")
        # layout output must be given
        
        output_detections = input_detections.copy()

        bboxes_only = [[item['bbox'] for item in page] for page in input_detections]

        res = self._get_text_detections(images, bboxes_only)

        for p, page in enumerate(res):
            for l, line in enumerate(page.text_lines):
                output_detections[p][l]['text'] = line.text
                output_detections[p][l]['text_confidence'] = line.confidence

        return output_detections
        
    def detect_text_lines_with_layout(
        self, 
        images: List[Image.Image],
        valid_layout_labels: List[str] = None,
        layout_confidence_threshold: float = 0.5,
        text_detection_confidence_threshold: float = 0.5, 
        min_text_line_iou_with_layout_roi: float = 0.7
    ) -> List[List[Dict[str, Any]]]:
        """
        Detects text lines within valid layout regions.
        Args:
            images: List of PIL Image objects.
            valid_layout_labels: List of layout labels to consider (e.g., ['Text', 'SectionHeader', 'ListItem']).
                                 If None, all layout boxes are considered (but filtering is still useful for position).
            layout_confidence_threshold: Minimum confidence for a layout box to be considered.
            text_detection_confidence_threshold: (May not apply directly to Surya's DetectionPredictor output)
            min_text_line_iou_with_layout_roi: Minimum IoU for a text line to be kept if it overlaps with a valid layout ROI.
        Returns:
            List of pages, each page a list of dictionaries:
            {'bbox': [text_line_x1, y1, x2, y2], 'label': layout_label, 'confidence': layout_confidence, 'position': layout_position_id}
        """
        if not self.detection_predictor and not self.layout_predictor:
            self.detection_predictor = DetectionPredictor(dtype=torch.float32) 
            self.layout_predictor = LayoutPredictor(dtype=torch.float32)
            
        if valid_layout_labels is None:
            valid_layout_labels = ['Text', 'SectionHeader', 'PageHeader', 'PageFooter', 
                                   'ListItem', 'Caption', 'Footnote', 'Title', 'TextInlineMath']

        all_pages_layout_structs = self._get_layout_predictions(images) # List[List[LayoutDict]]

        all_pages_raw_text_lines_results = self._get_raw_text_line_detections(images) # List[DetectionResult]

        final_pages_filtered_lines = []

        for page_idx, image in enumerate(images):
            layout_regions_on_page = all_pages_layout_structs[page_idx]
            raw_text_lines_on_page_result = all_pages_raw_text_lines_results[page_idx]
            
            # Filter layout regions by label and confidence
            valid_rois = []
            for region in layout_regions_on_page:
                if region['label'] in valid_layout_labels and region['confidence'] >= layout_confidence_threshold:
                    valid_rois.append(region) # region is {'bbox':..., 'label':..., 'confidence':..., 'position':...}
            
            if not valid_rois:
                logger.debug(f"Page {page_idx+1}: No valid layout ROIs found after filtering. No text lines will be extracted.")
                final_pages_filtered_lines.append([])
                continue

            # Sort ROIs by position (reading order hint from layout model)
            valid_rois.sort(key=lambda r: r['position'])

            page_filtered_text_lines = []
            
            if raw_text_lines_on_page_result and raw_text_lines_on_page_result.bboxes:
                # Store text lines with their original indices to prevent re-matching if already assigned
                text_lines_with_status = [{'bbox': [int(c) for c in tb.bbox], 
                                           'confidence': tb.confidence, # Confidence of the text line itself
                                           'matched_to_roi': False} 
                                          for tb in raw_text_lines_on_page_result.bboxes]

                for roi in valid_rois:
                    roi_bbox = roi['bbox']
                    for tl_idx, text_line_data in enumerate(text_lines_with_status):
                        if text_line_data['matched_to_roi']: # Already assigned to a (likely earlier) ROI
                            continue 
                        
                        text_line_bbox = text_line_data['bbox']
                        # Check IoU between text line and current ROI
                        iou = self._calculate_intersection_over_area(text_line_bbox, roi_bbox)

                        if iou >= min_text_line_iou_with_layout_roi:
                            # This text line is considered part of this ROI
                            page_filtered_text_lines.append({
                                'bbox': text_line_bbox,                                 # The precise text line bbox
                                'label': roi['label'],                                  # Label from the layout ROI it belongs to
                                'confidence': roi['confidence'],                        # Confidence of the LAYOUT region (or text_line_data['confidence'] for text line?)
                                                                                        # Decide which confidence is more relevant here or store both.
                                                                                        # For now, using layout ROI's confidence as it guided the selection.
                                'text_line_confidence': text_line_data['confidence'],   # Store actual text line confidence
                                'position': roi['position']                             # Inherit position from the ROI
                            })
                            text_lines_with_status[tl_idx]['matched_to_roi'] = True
            
            # Sort the final filtered lines by their inherited ROI position, then by y-coordinate, then x-coordinate
            page_filtered_text_lines.sort(key=lambda line: (line['position'], line['bbox'][1], line['bbox'][0]))
            final_pages_filtered_lines.append(page_filtered_text_lines)

            self.detection_predictor = None
            self.layout_predictor = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return final_pages_filtered_lines   

    def _get_text_detections(self, images: List[Image.Image], bboxes: List[List[List[int]]] = None) -> List[List[Any]]:
        if not self.recognition_predictor:
            self.recognition_predictor = RecognitionPredictor(dtype=torch.float32)
            
        with torch.no_grad():
            if not bboxes:
                predictions = self.recognition_predictor([image.convert("RGB") for image in images], det_predictor=self.detection_predictor, return_words=True, detection_batch_size= DETECTOR_BATCH_SIZE, recognition_batch_size= RECOGNITION_BATCH_SIZE)   # Ensure RGB and return words
            else:
                predictions = self.recognition_predictor([image.convert("RGB") for image in images], bboxes=bboxes, recognition_batch_size= RECOGNITION_BATCH_SIZE)   
            
            self.recognition_predictor = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if len(predictions) and hasattr(predictions[0], 'text_lines'):

            # print(predictions)
            return predictions                 # return a list of predictions (one for each page)
        return []
    
    #TODO: update to use the new workflow
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

    def _draw_on_image(self, image: Image.Image, items_to_draw: List[Dict[str, Any]],
                       draw_text_content: bool = False, default_box_color="lime", 
                       text_color="black", font_size=10, title_prefix="SuryaOCR"):
        img_display = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img_display)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        color_map = { # For different layout labels
            'Text': 'green', 'SectionHeader': 'blue', 'PageHeader': 'purple',
            'Picture': 'red', 'Table': 'orange', 'ListItem': 'cyan',
            'TextInlineMath': 'black', 
            # Add more as needed
        }

        for item in items_to_draw:
            if not isinstance(item.get('bbox'), list) or len(item['bbox']) != 4:
                continue # Skip items with invalid bbox

            x1, y1, x2, y2 = map(int, item['bbox'])
            box_color = color_map.get(item.get('label'), default_box_color)
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
            label_to_show = item.get('label', '')
            if item.get('confidence') is not None:
                label_to_show += f" ({item.get('confidence'):.2f})"
            if item.get('text_line_confidence') is not None: # For combined output
                label_to_show += f" TLConf:{item.get('text_line_confidence'):.2f}"


            if draw_text_content and item.get('text'):
                text_to_draw = item['text']
            elif label_to_show: # If not drawing text content, draw the label
                text_to_draw = label_to_show
            else:
                text_to_draw = None

            if text_to_draw:
                # Position text label slightly above the box
                text_position = (x1, y1 - font_size - 2 if y1 - font_size - 2 > 0 else y1 + 2)
                try: # textbbox can fail with some fonts/text
                    text_bbox_coords = draw.textbbox(text_position, text_to_draw, font=font)
                    draw.rectangle(text_bbox_coords, fill=box_color)
                    draw.text(text_position, text_to_draw, fill=text_color, font=font)
                except Exception as e_draw:
                    logger.warning(f"Could not draw text label '{text_to_draw}': {e_draw}")
        
        img_display.show(title=f"{title_prefix} Output")

    def display_layout_regions(self, image: Image.Image):
        """Displays detected layout regions on the image."""
        logger.info("SuryaOCR: Displaying layout regions.")
        layout_data_per_page = self._get_layout_predictions([image.copy()])
        if layout_data_per_page:
            self._draw_on_image(image, layout_data_per_page[0], draw_text_content=False, title_prefix="SuryaLayout")
        else:
            logger.warning("No layout data to display.")
            image.show(title="SuryaLayout Output (No Detections)")

    def display_detected_text_lines(self, image: Image.Image, with_layout_filtering: bool = True):
        """Displays detected text lines, optionally filtered by layout."""
        logger.info(f"SuryaOCR: Displaying detected text lines {'with' if with_layout_filtering else 'without'} layout filtering.")
        if with_layout_filtering:
            lines_data_per_page = self.detect_text_lines_with_layout([image.copy()])
            title = "Surya Text Lines (Layout Filtered)"
        else:
            raw_detections_per_page = self._get_raw_text_line_detections([image.copy()]) # List[DetectionResult]
            lines_data_per_page = []
            if raw_detections_per_page and raw_detections_per_page[0].bboxes:
                lines_data_per_page.append(
                    [{'bbox': [int(c) for c in tb.bbox], 'confidence': tb.confidence, 'label': 'TextLine'} 
                     for tb in raw_detections_per_page[0].bboxes]
                )
            title = "Surya Text Lines (Raw)"

        if lines_data_per_page and lines_data_per_page[0]:
            self._draw_on_image(image, lines_data_per_page[0], draw_text_content=False, title_prefix=title)
        else:
            logger.warning("No text line data to display.")
            image.show(title=f"{title} (No Detections)")

    #TODO: update to new pipeline
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
    def _calculate_intersection_over_area(self, box_target: List[float], box_reference: List[float]) -> float:
        """
        Calculates Intersection(box_target, box_reference) / Area(box_target).
        Useful for checking how much of box_target is within box_reference.
        Boxes are expected in [x1, y1, x2, y2] format.
        """
        if not (isinstance(box_target, (list, tuple)) and len(box_target) == 4 and
                isinstance(box_reference, (list, tuple)) and len(box_reference) == 4):
            logger.warning(f"Invalid bounding box format for IoA. BoxTarget: {box_target}, BoxRef: {box_reference}")
            return 0.0
        try:
            # Determine the (x, y)-coordinates of the intersection rectangle
            xA = max(box_target[0], box_reference[0])
            yA = max(box_target[1], box_reference[1])
            xB = min(box_target[2], box_reference[2])
            yB = min(box_target[3], box_reference[3])

            inter_width = xB - xA
            inter_height = yB - yA

            if inter_width <= 0 or inter_height <= 0:
                return 0.0  # No overlap

            interArea = inter_width * inter_height
            box_target_Area = (box_target[2] - box_target[0]) * (box_target[3] - box_target[1])

            if box_target_Area <= 0:
                # logger.warning(f"Invalid target box area: {box_target_Area} for box {box_target}")
                return 0.0

            ioa = interArea / float(box_target_Area)
            return max(0.0, min(ioa, 1.0)) # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error in _calculate_intersection_over_area: Target: {box_target}, Ref: {box_reference} - {e}")
            return 0.0