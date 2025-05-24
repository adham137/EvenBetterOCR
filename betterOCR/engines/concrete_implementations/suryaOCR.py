from typing import List
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

import torch, os
from engines.IEngine import OCREngine
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

os.environ['COMPILE_LAYOUT']='true'
os.environ['LAYOUT_BATCH_SIZE']='16'

class SuryaOCREngine(OCREngine):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init detection and recognition models
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

        self.detection_predictor.model = self.detection_predictor.model.to(device=device, dtype=torch.float32)
        # recognition_predictor.model = recognition_predictor.model.to(device=device, dtype=torch.float32)

        # self.layout_predictor = LayoutPredictor()
        # self.layout_predictor.model = self.layout_predictor.model.to(device=device, dtype=torch.float32)
        
    def execute(self, image):
        return self.detection_predictor([image])
    

    def __detect_boxes(self, images: List[Image.Image]) -> List:
        return self.detection_predictor(images)

    
    def draw_detection_boxes(self, image,
                         box_color="lime", 
                         text_color="black", 
                         font_size=14):
        """
    Draw detection bboxes with confidence scores on the image and display it.

    Args:
        image (PIL.Image.Image): The original image.
        box_color (str): Color for the bounding box lines.
        text_color (str): Color for the text labels.
        font_size (int): Font size for confidence labels.
    """
        det_results = self.__detect_boxes([image])
        # Make a copy and get a drawing context
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)

        # Try to load a truetype font; fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Iterate over each detection result
        for result in det_results:
            for det in result.bboxes:
                x0, y0, x1, y1 = det.bbox
                conf = det.confidence

                # Draw the bounding box
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)

                # Prepare label text
                text = f"{conf:.2f}"
                bbox = draw.textbbox((x0, y0), text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Draw background for text
                draw.rectangle(
                    [x0, y0 - text_h - 4, x0 + text_w + 4, y0], 
                    fill=box_color
                )
                # Draw the text itself
                draw.text((x0 + 2, y0 - text_h - 2), text, fill=text_color, font=font)

        # Display with matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()