

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


PDF_PATH = 'data\\table_in_page_2.pdf'#'data\\alamiria_2003_88.pdf'
parser = DocumentParser()
images = parser.load_images_from_document(PDF_PATH)

# tesseractOCR = TesseractOCREngine(['ar'])
# out = tesseractOCR.recognize_text(images)
# print('******************\n'.join([ page for page in out]))
# tesseractOCR.display_bounding_boxes(images[0])
# tesseractOCR.display_annotated_output(images[1])
 
# eOCR = EasyOCREngine(['ar'], gpu=True)
# eOCR.display_annotated_output(image_1)
# eOCR.display_bounding_boxes(image_1)

sOCR = SuryaOCREngine(['ar'])
temp = sOCR.detect_text_lines_with_layout(images) ## Processing of 50 pages took appx 30 sec 
#[
#   // page_1
#   [
#       {
#           bbox: [147, 137, 448, 152]
#           label: 'PageHeader'
#           confidence: 0.99
#           text_line_confidence: 0.98
#           position: 0
#       },
#       {obj_2}
#   ],
#   
#   [page_2]
#]
list_og = [obj['bbox'] for obj in temp[0]]

list_h = [[coord[0], coord[2], coord[1], coord[3]] for coord in list_og]
# free_list is a list of free-form text boxes. The format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
list_f = [ [ [coords[0], coords[1]], [coords[0], coords[3]], [coords[2], coords[3]], [coords[2], coords[1]]] for coords in list_og]

reader = easyocr.Reader(['ar'], detector=False)
img_np = np.array(images[0].convert("RGB"))
eOCR = reader.recognize(img_np, horizontal_list=list_h, free_list= list_f)
print(eOCR)
# sOCR.display_detected_text_lines(images[1])

# sOCR.recognize_text(images)
# sOCR.display_textline_boxes(images[1])
# structured_output = sOCR.get_structured_output(images)
# sOCR.display_annotated_output(images[1], structured_output[1])
# sOCR.display_bounding_boxes(images[0], structured_output[0])


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