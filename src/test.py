

import json
import pprint
from PIL import Image
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from engines.concrete_implementations.easyOCR import EasyOCREngine
from engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from engines.concrete_implementations.suryaOCR import SuryaOCREngine
from parsers.parser import DocumentParser


PDF_PATH = 'data\\table_in_page_2.pdf'
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
sOCR.display_textline_boxes(images[1])
structured_output = sOCR.get_structured_output(images)
sOCR.display_annotated_output(images[1], structured_output[1])
# sOCR.display_bounding_boxes(images[0], structured_output[0])
