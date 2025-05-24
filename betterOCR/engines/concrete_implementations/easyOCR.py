import easyocr
from engines.IEngine import OCREngine

class EasyOCREngine(OCREngine):
    def __init__(self, lang):
        self.reader = easyocr.Reader(lang)
        
    def execute(self, image):
        return self.reader.readtext(image, detail=0)
    
#     def detect_boxes(self, image_path):
#         return self.reader.readtext(image_path, output_format="dict")
    
#     def detect_croped_text(self, image, layout_results):
#         for result in layout_results:
#             for box in result.bboxes:
#                 x0, y0, x1, y1 = box.bbox
#                 crop = image[y0:y1, x0:x1]
#                 crop_result = self.execute(crop)
#                 print(crop_result)


# # engine_registry.register_engine("easyocr", EasyOCREngine)