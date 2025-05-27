

import json
import pprint
from PIL import Image
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from engines.concrete_implementations.easyOCR import EasyOCREngine
from engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from engines.concrete_implementations.suryaOCR import SuryaOCREngine

IMAGE_PATH_1 = 'D:\\ASU\\sem 10\\GRAD PROJ\\EvenBetterOCR\\test\\input\\page_1.png'
IMAGE_PATH_2 = 'C:\\Users\\Adham\\Downloads\\Ameriya_Extract\\0 (1).png'
IMAGE_PATH_3 = 'C:\\Users\\Adham\\Downloads\\Ameriya_Extract\\0 (2).png'
image_1 = Image.open(IMAGE_PATH_1)
image_2 = Image.open(IMAGE_PATH_2)
image_3 = Image.open(IMAGE_PATH_3)

# eOCR = EasyOCREngine(['ar'], gpu=True)
# eOCR.display_annotated_output(image_1)
# eOCR.display_bounding_boxes(image_1)

tesseract = TesseractOCREngine(['ar'])
# tesseract.display_bounding_boxes(image_1)
tesseract.display_annotated_output(image_1)

# sOCR = SuryaOCREngine(['ar'])
# sOCR.display_textline_boxes(image_1)
# structured_output = sOCR.get_structured_output(image_1)
# sOCR.display_annotated_output(image_1, structured_output)
# sOCR.display_bounding_boxes(image_1, structured_output)
