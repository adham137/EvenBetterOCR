

import json
import pprint
from typing import List
from PIL import Image
import easyocr
import numpy as np
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from src.engines.concrete_implementations.easyOCR import EasyOCREngine
from src.engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from src.engines.concrete_implementations.suryaOCR import SuryaOCREngine
from src.parsers.parser import DocumentParser

from src.llm.clients.gemini_client import GeminiClient


# PDF_PATH = 'D:\\ASU\\sem 10\\GRAD PROJ\\Getting Data\\merged.pdf'#'data\\table_in_page_2.pdf'#'data\\alamiria_2003_88.pdf'
# parser = DocumentParser()
# images = parser.load_images_from_document(PDF_PATH)

# sOCR = SuryaOCREngine(['ar'])
# # temp = sOCR.detect_text_lines_with_layout(images) ## Processing of 50 pages took appx 30 sec 

# sOCR.display_detected_text_lines(images[10])


# tesseractOCR = TesseractOCREngine(['ar'])
# tesseract_recognized_pages = tesseractOCR.recognize_detected_lines(
#     images,
#     temp # Output from your Surya detector (or any other detector)
# )
# print(tesseract_recognized_pages)

# out = tesseractOCR.recognize_text(images)
# print('******************\n'.join([ page for page in out]))
# tesseractOCR.display_bounding_boxes(images[0])
# tesseractOCR.display_annotated_output(images[1])
 
# eOCR = EasyOCREngine(['ar'], gpu=True)
# eOCR.display_annotated_output(image_1)
# eOCR.display_bounding_boxes(image_1)



# temp =
# [
#   // page_1
#   [
#       {   // obj_1
#           bbox: [147, 137, 448, 152]
#           label: 'PageHeader'
#           confidence: 0.99
#           text_line_confidence: 0.98
#           position: 0
#       },
#       {obj_2}, ...
#   ],
#   
#   [page_2], ...
#]

# surya_recognized_pages = sOCR.get_structured_output(images, input_detections = temp )
# print(surya_recognized_pages)
# [
#   // page_1
#   [
#       {   // obj_1
#           bbox: [147, 137, 448, 152]
#           label: 'PageHeader'
#           confidence: 0.99
#           text_line_confidence: 0.98
#           position: 0
#           text: '...'
#           text_confidence: 0.97
#       },
#
#       {obj_2} , ...
#   ],
#   
#   [page_2] , ...
#]

# sOCR.display_detected_text_lines(images[1])

# sOCR.recognize_text(images)
# sOCR.display_textline_boxes(images[1])
# structured_output = sOCR.get_structured_output(images)
# sOCR.display_annotated_output(images[1], structured_output[1])
# sOCR.display_bounding_boxes(images[0], structured_output[0])

# from src.combiner.lineMerger import LineROVERMerger
# lm = LineROVERMerger()
# out = lm.merge_document_results(surya_recognized_pages, tesseract_recognized_pages)
# print(out)
# from database.app.repositories.document_repository import DocumentRepository
# dr = DocumentRepository()
# results = []
# with open('data\\ocr_output.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
# for pdf_name, pages in data.items():
#     results.append(dr.insert_chunked_document(pages, pdf_name))
# print(results)

# recognition_predictor = RecognitionPredictor()
# detection_predictor = DetectionPredictor(dtype=torch.float32)

# first_param = next(detection_predictor.model.parameters(), None)
# if first_param is None:
#     print("Model detec has no parameters!")
# else:
#     print(f"Model detec is on device: {first_param.device}")
# first_param = next(recognition_predictor.model.parameters(), None)
# if first_param is None:
#     print("Model recogn has no parameters!")
# else:
#     print(f"Model recogn is on device: {first_param.device}")

# res = recognition_predictor([image.convert("RGB") for image in images], det_predictor=detection_predictor, return_words=True)
# print(res)