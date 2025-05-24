# import easyocr
# import cv2
# import numpy as np
# from typing import List, Dict, Tuple, Union, Any
# from PIL import Image
# from engines.concrete_implementations.suryaOCR import SuryaOCREngine

# def extract_layout_boxes(surya_output) -> List[Dict]:
#     """
#     Extract layout boxes from Surya OCR output object.
    
#     Args:
#         surya_output: Object output from Surya OCR
        
#     Returns:
#         List of dictionaries containing information about each detected layout element
#     """
#     results = []
    
#     # Check if the output contains LayoutResult with bboxes
#     if not hasattr(surya_output, '__iter__'):
#         # Try to extract from a single LayoutResult object
#         if hasattr(surya_output, 'bboxes'):
#             boxes = surya_output.bboxes
#         else:
#             raise ValueError("Unsupported Surya output format")
#     else:
#         # Handle when surya_output is a list of LayoutResults
#         boxes = []
#         for layout_result in surya_output:
#             if hasattr(layout_result, 'bboxes'):
#                 boxes.extend(layout_result.bboxes)
    
#     # Process each layout box
#     for box in boxes:
#         results.append({
#             'polygon': box.polygon,
#             'confidence': box.confidence,
#             'label': box.label,
#             'bbox': box.bbox,
#             'position': box.position if hasattr(box, 'position') else None
#         })
            
#     return results

# def filter_text_regions(layout_boxes: List[Dict]) -> List[Dict]:
#     """
#     Filter layout boxes to include only text regions (excluding images and handwriting).
    
#     Args:
#         layout_boxes: List of parsed layout boxes from Surya
        
#     Returns:
#         Filtered list containing only text regions
#     """
#     excluded_labels = ['Picture', 'Handwriting']
    
#     return [
#         box for box in layout_boxes 
#         if box['label'] not in excluded_labels
#     ]

# def extract_roi_from_image(image: np.ndarray, bbox: List[float]) -> np.ndarray:
#     """
#     Extract a region of interest from an image using a bounding box.
    
#     Args:
#         image: Input image
#         bbox: Bounding box coordinates [x1, y1, x2, y2]
        
#     Returns:
#         Cropped image containing only the region of interest
#     """
#     x1, y1, x2, y2 = map(int, bbox)
#     img_croped = image[y1:y2, x1:x2]
#     img_resized = cv2.resize(img_croped, (0, 0), fx=1, fy=1)
#     return img_resized

# def process_with_easyocr(image_path: str, surya_output) -> List[Dict]:
#     """
#     Process an image with easyOCR using only text regions identified by Surya OCR.
    
#     Args:
#         image_path: Path to the input image
#         surya_output: Object output from Surya OCR
        
#     Returns:
#         List of dictionaries containing text recognition results for each text region
#     """
#     # Initialize easyOCR reader
#     reader = easyocr.Reader(['ar'])
    
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not load image from {image_path}")
    
#     # Extract layout boxes from Surya output
#     layout_boxes = extract_layout_boxes(surya_output)
    
#     # Filter to get only text regions
#     text_regions = filter_text_regions(layout_boxes)
    
#     results = []
#     for i, region in enumerate(text_regions):
#         bbox = region['bbox']
        
#         # Extract region of interest
#         roi = extract_roi_from_image(image, bbox)
        
#         # Process with easyOCR
#         ocr_result = reader.readtext(roi)
        
#         results.append({
#             'region_id': i,
#             'label': region['label'],
#             'bbox': bbox,
#             'confidence': region['confidence'],
#             'ocr_result': ocr_result
#         })
    
#     return results

# def draw_annotations(image: np.ndarray, surya_boxes: List[Dict], ocr_results: List[Dict]) -> np.ndarray:
#     """
#     Draw annotations on the image showing Surya bounding boxes and easyOCR text detections.
    
#     Args:
#         image: Original image as numpy array
#         surya_boxes: List of Surya layout boxes
#         ocr_results: List of easyOCR results
        
#     Returns:
#         Annotated image with bounding boxes
#     """
#     # Create a copy of the image to draw on
#     annotated_img = image.copy()
    
#     # Draw Surya layout boxes
#     for box in surya_boxes:
#         # Get coordinates
#         x1, y1, x2, y2 = map(int, box['bbox'])
        
#         # Determine color based on label (different colors for different types)
#         if box['label'] == 'Text':
#             color = (0, 255, 0)  # Green for text
#         elif box['label'] == 'SectionHeader':
#             color = (0, 0, 255)  # Blue for section headers
#         elif box['label'] == 'Picture':
#             color = (255, 0, 0)  # Red for pictures
#         elif box['label'] == 'Handwriting':
#             color = (255, 255, 0)  # Cyan for handwriting
#         else:
#             color = (128, 128, 128)  # Gray for other elements
        
#         # Draw rectangle for Surya box
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
#         # # Add label text
#         # cv2.putText(annotated_img, box['label'], (x1, y1-10), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     # Draw easyOCR text detections
#     for result in ocr_results:
#         # Get region coordinates
#         x1, y1, x2, y2 = map(int, result['bbox'])
        
#         # Loop through each text detection in this region
#         for i, (text_bbox, text, conf) in enumerate(result['ocr_result']):
#             # Convert easyOCR bbox format to x1,y1,x2,y2
#             # easyOCR returns points in format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
#             # We need to offset them by the region position
#             if len(text_bbox) == 4:  # If it's a quadrilateral
#                 pts = np.array(text_bbox, np.int32)
#                 # Add region offset
#                 pts[:, 0] += x1
#                 pts[:, 1] += y1
#                 # Draw polygon for text area
#                 cv2.polylines(annotated_img, [pts], True, (0, 165, 255), 1)
                
#                 # Add detected text near the box
#                 text_position = (pts[0][0], pts[0][1] - 5)
#                 cv2.putText(annotated_img, text[:20], text_position, 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
#     return annotated_img

# def combine_ocr_text(ocr_results: List[Dict]) -> str:
#     """
#     Combine all text detections from easyOCR results into a single string,
#     maintaining the order of regions as they appear in the document.
    
#     Args:
#         ocr_results: List of easyOCR results per region
        
#     Returns:
#         Combined text from all regions
#     """
#     # Sort results by position (top to bottom)
#     sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
    
#     combined_text = ""
    
#     for result in sorted_results:
#         # For each text region
#         region_text = []
#         for _, text, _ in result['ocr_result']:
#             region_text.append(text)
        
#         # Join all text from this region
#         if region_text:
#             combined_text += " ".join(region_text) + "\n\n"
    
#     return combined_text

# def direct_surya_to_easyocr(image_path: str, surya_output, save_annotated=True, output_txt=True) -> Dict:
#     """
#     Main function to directly process Surya OCR output with easyOCR.
    
#     Args:
#         image_path: Path to the input image
#         surya_output: Surya OCR output object
#         save_annotated: Whether to save the annotated image
#         output_txt: Whether to output the combined text as a file
        
#     Returns:
#         Dictionary containing OCR results, paths to output files, and combined text
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not load image from {image_path}")
    
#     # Extract layout boxes from Surya output
#     layout_boxes = extract_layout_boxes(surya_output)
    
#     # Get all text regions
#     text_regions = filter_text_regions(layout_boxes)
    
#     # Process with easyOCR
#     ocr_results = process_with_easyocr(image_path, surya_output)
    
#     # Draw annotations
#     annotated_image = draw_annotations(image, layout_boxes, ocr_results)
    
#     # Output files
#     output_files = {}
    
#     # Save annotated image
#     if save_annotated:
#         annotated_path = image_path.rsplit('.', 1)[0] + '_annotated.' + image_path.rsplit('.', 1)[1]
#         cv2.imwrite(annotated_path, annotated_image)
#         output_files['annotated_image'] = annotated_path
    
#     # Combine all OCR text
#     combined_text = combine_ocr_text(ocr_results)
    
#     # Save combined text to file
#     if output_txt:
#         txt_path = image_path.rsplit('.', 1)[0] + '_ocr.txt'
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(combined_text)
#         output_files['text_file'] = txt_path
    
#     return {
#         'ocr_results': ocr_results,
#         'output_files': output_files,
#         'combined_text': combined_text,
#         'annotated_image': annotated_image if save_annotated else None
#     }

# def main():
#     """
#     Example usage of the OCR processing pipeline.
#     """
#     import sys
#     from ast import literal_eval
    
#     # Example paths
#     image_path = "D:\\ASU\\sem 10\\GRAD PROJ\\EvenBetterOCR\\test\\input\\page_1.png"
#     image = Image.open(image_path)
#     surya = SuryaOCREngine('ar')
#     surya_output = surya.execute(image)
    
#     # Process image with easyOCR using Surya regions and save outputs
#     results = direct_surya_to_easyocr(image_path, surya_output, save_annotated=True, output_txt=True)
    
#     # Print results
#     print(f"OCR processing complete:")
#     print(f"- Annotated image saved to: {results['output_files'].get('annotated_image', 'Not saved')}")
#     print(f"- OCR text file saved to: {results['output_files'].get('text_file', 'Not saved')}")
#     print(f"\nExtracted text preview:")
#     print(results['combined_text'][:500] + "..." if len(results['combined_text']) > 500 else results['combined_text'])

# if __name__ == "__main__":
#     main()
from PIL import Image
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from engines.concrete_implementations.easyOCR import EasyOCREngine
from engines.concrete_implementations.tesseractOCR import TesseractOCREngine

IMAGE_PATH_1 = 'D:\\ASU\\sem 10\\GRAD PROJ\\EvenBetterOCR\\test\\input\\page_1.png'
IMAGE_PATH_2 = 'C:\\Users\\Adham\\Downloads\\Ameriya_Extract\\0 (1).png'
IMAGE_PATH_3 = 'C:\\Users\\Adham\\Downloads\\Ameriya_Extract\\0 (2).png'
image_1 = Image.open(IMAGE_PATH_1)
image_2 = Image.open(IMAGE_PATH_2)
image_3 = Image.open(IMAGE_PATH_3)

# eOCR = EasyOCREngine(['ar'], gpu=True)
# eOCR.display_annotated_output(image_1)
# eOCR.display_bounding_boxes(image_1)

# tesseract = TesseractOCREngine(['ara'])
# tesseract.display_bounding_boxes(image_1)
# tesseract.display_annotated_output(image_1)
