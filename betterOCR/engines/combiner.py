
import threading
import logging
from typing import List, Dict, Type
from PIL import Image

from engines.IEngine import OCREngine
from engines.EngineRegistry import EngineRegistry # To get engine classes
from parsers.parser import DocumentParser

logger = logging.getLogger(__name__)

class OCRCombiner:
    def __init__(self, 
                 engine_registry: EngineRegistry,
                 engine_names: List[str],
                 document_parser: DocumentParser,
                 document_path: str,
                 lang: List[str], 
                 engine_configs: Dict[str, Dict] = None):
        """
        Initializes the OCRCombiner.
        Args:
            engine_registry: Instance of EngineRegistry to fetch engine classes.
            engine_names: List of names of OCR engines to use (e.g., ["easyocr", "tesseractocr"]).
            document_parser: Instance of DocumentParser.
            document_path: Path to the document to be processed.
            lang: List of language codes for OCR engines.
            engine_configs: Optional dictionary with specific configurations for each engine.
                            Format: {"engine_name": {"param1": "value1", ...}}
        """
        self.engine_registry = engine_registry
        self.engine_names = engine_names
        self.document_parser = document_parser
        self.document_path = document_path
        self.lang = lang
        self.engine_configs = engine_configs if engine_configs else {}
        logger.info(f"OCRCombiner initialized for engines: {engine_names} on document: {document_path}")

    def _run_single_engine_on_image(self, engine_class: Type[OCREngine], image: Image.Image, lang: List[str], config: Dict, results_list: list, engine_name: str):
        """Target function for threading to run one engine on one image."""
        try:
            engine_instance = engine_class(lang=lang, **config) 
            logger.info(f"Thread for {engine_name}: Starting OCR on image...")
            text_output = engine_instance.recognize_text(image)
            results_list.append({"engine": engine_name, "text": text_output})
            logger.info(f"Thread for {engine_name}: Finished OCR. Text length: {len(text_output)}")
        except Exception as e:
            logger.error(f"Thread for {engine_name}: Error during OCR processing: {e}")
            results_list.append({"engine": engine_name, "text": f"[ERROR: {engine_name} failed - {e}]"})

    def run_ocr_pipeline_parallel(self) -> str:
        """
        Processes the document using multiple OCR engines in parallel (per image)
        and combines their text outputs.
        Returns:
            A single string which is a concatenation of all OCR outputs.
            Format:
            --- Page 1 ---
            [Engine1 Output Page 1]
            [Engine2 Output Page 1]
            --- Page 2 ---
            [Engine1 Output Page 2]
            ...
        """
        logger.info(f"Starting OCR pipeline for document: {self.document_path}")
        try:
            images = self.document_parser.load_images_from_document(self.document_path)
        except Exception as e:
            logger.error(f"Failed to load images for OCR pipeline: {e}")
            return f"[ERROR: Failed to load document - {e}]"

        if not images:
            logger.warning("No images found or loaded from the document.")
            return "[ERROR: No images to process]"

        final_combined_text = []
        
        for i, image in enumerate(images):
            page_number = i + 1
            logger.info(f"Processing Page {page_number}/{len(images)}")
            page_results_collector = [] # [(engine_name, text), ...] 
            threads = []

            for engine_name in self.engine_names:
                try:
                    engine_class = self.engine_registry.get_engine_class(engine_name) 
                    specific_config = self.engine_configs.get(engine_name, {})
                    
                    thread = threading.Thread(
                        target=self._run_single_engine_on_image,
                        args=(engine_class, image.copy(), self.lang, specific_config, page_results_collector, engine_name)
                    )
                    threads.append(thread)
                    thread.start()
                    logger.debug(f"Thread started for {engine_name} on Page {page_number}.")
                except ValueError as ve: 
                    logger.error(f"Could not get engine '{engine_name}': {ve}")
                    page_results_collector.append({"engine": engine_name, "text": f"[ERROR: Engine {engine_name} not found or failed to initialize]"})
                except Exception as e:
                     logger.error(f"Failed to start thread for engine '{engine_name}' on Page {page_number}: {e}")
                     page_results_collector.append({"engine": engine_name, "text": f"[ERROR: Could not start {engine_name} - {e}]"})


            for thread in threads:
                thread.join()
            
            logger.info(f"All OCR engines finished for Page {page_number}.")
            
            # Combine results for the current page
            # Ensure consistent order of engine outputs if necessary, though threads don't guarantee order of append.
            # For simplicity, we'll just take them as they finished, or sort by engine name for consistency.
            page_results_collector.sort(key=lambda x: self.engine_names.index(x['engine']) if x['engine'] in self.engine_names else -1)

            final_combined_text.append(f"--- Page {page_number} ---\n")
            for res in page_results_collector:
                final_combined_text.append(f"--- OCR Output from {res['engine']} for Page {page_number} ---\n{res['text']}\n\n")
            
        logger.info("OCR pipeline completed for all pages.")
        return "".join(final_combined_text)