
import threading
import logging
from typing import Any, List, Dict, Type
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
                 lang_list: List[str], 
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
        self.lang_list = lang_list
        self.engine_configs = engine_configs if engine_configs else {}
        logger.info(f"OCRCombiner initialized for engines: {engine_names} on document: {document_path}")

    def _run_engine_on_all_images(
        self,
        engine_class: Type[OCREngine],
        all_images: List[Image.Image],
        lang_list_for_engine: List[str],
        config: Dict[str, Any],
        results_dict: Dict[str, List[str]], # engine_name -> list of texts (one per page)
        engine_name: str
    ) -> None:
        """
        Target function for a thread. Runs one OCR engine on ALL images (document pages)
        and stores its list of recognized texts (or error placeholders) into a shared dictionary.

        Parameters:
            engine_class:    The OCR engine class to instantiate.
            all_images:           A list of PIL Images to run OCR on (each thread gets its own copy).
            lang_list_for_engine:            List of language codes for the engine constructor.
            config:          Engine-specific config dict (e.g. tesseract parameters).
            results_list:    A thread-safe dictionary to which we append:
                            {engine_name : [page_1_text, page_2_text, etc.]}
            engine_name:     A string key identifying this engine (must match self.engine_names).
        """
        try:
            engine_instance = engine_class(lang_list=lang_list_for_engine, **config)
            logger.info(f"[{engine_name}] Starting OCR on image(s)…")
            text_outputs = engine_instance.recognize_text(all_images)
            results_dict[engine_name] = text_outputs
            logger.info(f"[{engine_name}] Finished OCR; Number of pages processed= {len(all_images)}")
        except Exception as e:
            logger.error(f"[{engine_name}] Error during OCR: {e}")
            results_dict[engine_name] = [f"[ERROR: {engine_name} failed – {e}]" for _ in range(len(all_images))]


    def run_ocr_pipeline_parallel(self) -> List[List[str]]:
        """
        Processes the document using multiple OCR engines in parallel. Each engine
        processes all pages of the document. The results are then collated page by page.

        Returns:
            List of pages, where each page is itself a list of strings:
                [
                [  # page 1
                    text_from_engine_0_on_page_1,
                    text_from_engine_1_on_page_1,
                    ...
                ],
                [  # page 2
                    text_from_engine_0_on_page_2,
                    text_from_engine_1_on_page_2,
                    ...
                ],
                ...
                ]

        Notes:
            - The inner list is ordered exactly as `self.engine_names`.
            - Any engine that fails to initialize, start, or recognize will contribute
            a single string of the form "[ERROR: <engine_name> ...]".
        """
        logger.info(f"Starting OCR pipeline for document: {self.document_path}")

        try:
            all_document_images = self.document_parser.load_images_from_document(self.document_path)
        except Exception as e:
            logger.error(f"Failed to load document pages: {e}")
            return [[f"[ERROR: Failed to load document – {e}]"]]

        if not all_document_images:
            logger.warning("No images found in document.")
            return [["[ERROR: No images to process]"]]
        
        num_pages = len(all_document_images)
        logger.info(f"Document loaded with {num_pages} pages.")

        engine_results_map: Dict[str, List[str]] = {} # engine_name -> List[text_for_page_i]
        threads: List[threading.Thread] = []

        for engine_name in self.engine_names:
            try:
                engine_cls = self.engine_registry.get_engine_class(engine_name)
                cfg = self.engine_configs.get(engine_name, {})
                
                t = threading.Thread(
                    target=self._run_engine_on_all_images,
                    args=(engine_cls, all_document_images, self.lang_list, cfg, # Pass self.lang_list
                          engine_results_map, engine_name)
                )
                threads.append(t)
                t.start()
                logger.debug(f"Started thread for {engine_name} to process all {num_pages} pages.")
            except Exception as e:
                logger.error(f"Could not start OCR thread for '{engine_name}': {e}", exc_info=True)
                engine_results_map[engine_name] = [f"[ERROR: Could not start thread for {engine_name} – {e}]" for _ in range(num_pages)]

        for t in threads:
            t.join()
        logger.info(f"All engine threads finished processing all pages.")

        all_pages_outputs: List[List[str]] = []
        for page_idx in range(num_pages):
            page_texts: List[str] = []
            for engine_name in self.engine_names: # Ensure consistent order based on self.engine_names
                engine_page_results_list = engine_results_map.get(engine_name)
                if engine_page_results_list and page_idx < len(engine_page_results_list):
                    page_texts.append(engine_page_results_list[page_idx])
                else:
                    logger.warning(f"Missing or incomplete result for engine '{engine_name}' on page {page_idx + 1}. Using placeholder.")
                    page_texts.append(f"[ERROR: Result unavailable for {engine_name}, page {page_idx + 1}]")
            all_pages_outputs.append(page_texts)
        
        logger.info("OCR pipeline completed. Results collated for all pages.")
        return all_pages_outputs