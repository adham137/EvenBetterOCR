from PIL import Image
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        logger.info("DocumentParser initialized.")

    def load_images_from_document(self, document_path: str) -> List[Image.Image]:
        """
        Loads all pages from a document (PDF or image file) as PIL Image objects.
        Args:
            document_path: Path to the document file.
        Returns:
            A list of PIL Image objects, one for each page.
        Raises:
            FileNotFoundError: If the document_path does not exist.
            ValueError: If the document type is unsupported or loading fails.
        """
        if not os.path.exists(document_path):
            logger.error(f"Document not found at path: {document_path}")
            raise FileNotFoundError(f"Document not found: {document_path}")

        _, extension = os.path.splitext(document_path.lower())
        images = []

        try:
            if extension == '.pdf':
                logger.debug(f"Parsing PDF document: {document_path}")
                from pdf2image import convert_from_path
                images = convert_from_path(document_path)
                logger.info(f"Successfully parsed {len(images)} pages from PDF: {document_path}")
            elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                logger.debug(f"Loading image document: {document_path}")
                img = Image.open(document_path)
                images.append(img)
                logger.info(f"Successfully loaded image: {document_path}")
            else:
                logger.warning(f"Unsupported document extension: {extension} for file: {document_path}")
                raise ValueError(f"Unsupported document type: {extension}")
        except ImportError:
            if extension == '.pdf':
                logger.error("pdf2image library is not installed. Please install it to process PDF files ('pip install pdf2image').")
                raise ImportError("pdf2image is required for PDF processing.")
        except Exception as e:
            logger.error(f"Failed to load or parse document {document_path}: {e}")
            raise ValueError(f"Could not process document {document_path}: {e}")
            
        return images