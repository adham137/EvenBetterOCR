
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

    def _run_single_engine_on_image(
        self,
        engine_class: Type[OCREngine],
        image: Image.Image,
        lang: List[str],
        config: Dict[str, Any],
        results_list: List[Dict[str, str]],
        engine_name: str
    ) -> None:
        """
        Target function for a thread which runs one OCR engine on one image 
        and appends its raw text (or an error placeholder) into a shared list.

        Parameters:
            engine_class:    The OCR engine class to instantiate.
            image:           A PIL Image to run OCR on (each thread gets its own copy).
            lang:            List of language codes for the engine constructor.
            config:          Engine-specific config dict (e.g. tesseract parameters).
            results_list:    A thread-safe list to which we append:
                            {"engine": engine_name, "text": recognized_text}
            engine_name:     A string key identifying this engine (must match self.engine_names).
        """
        try:
            engine_instance = engine_class(lang=lang, **config)
            logger.info(f"[{engine_name}] Starting OCR on image…")
            text_output = engine_instance.recognize_text(image)
            results_list.append({"engine": engine_name, "text": text_output})
            logger.info(f"[{engine_name}] Finished OCR; text length={len(text_output)}")
        except Exception as e:
            logger.error(f"[{engine_name}] Error during OCR: {e}")
            results_list.append({
                "engine": engine_name,
                "text": f"[ERROR: {engine_name} failed – {e}]"
            })


    def run_ocr_pipeline_parallel(self) -> List[List[str]]:
        """
        Processes the document using multiple OCR engines in parallel, page by page,
        and returns a nested list of raw text outputs.

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
            - Engines run in their own threads so that page-level processing is parallel
            across engines, but pages themselves are handled sequentially.
            - The inner list is ordered exactly as `self.engine_names`.
            - Any engine that fails to initialize, start, or recognize will contribute
            a single string of the form "[ERROR: <engine_name> ...]".
        """
        logger.info(f"Starting OCR pipeline for document: {self.document_path}")

        try:
            images = self.document_parser.load_images_from_document(self.document_path)
        except Exception as e:
            logger.error(f"Failed to load document pages: {e}")
            # Return a single page with a single ERROR string
            return [[f"[ERROR: Failed to load document – {e}]"]]

        if not images:
            logger.warning("No images found in document.")
            return [["[ERROR: No images to process]"]]

        all_pages_outputs: List[List[str]] = []

        for page_idx, image in enumerate(images):
            page_number = page_idx + 1
            logger.info(f"Processing Page {page_number}/{len(images)}")

            # Collector for dicts {"engine": name, "text": ...}
            page_results_collector: List[Dict[str, str]] = []
            threads: List[threading.Thread] = []

            # Spin up one thread per engine
            for engine_name in self.engine_names:
                try:
                    engine_cls = self.engine_registry.get_engine_class(engine_name)
                    cfg = self.engine_configs.get(engine_name, {})
                    t = threading.Thread(
                        target=self._run_single_engine_on_image,
                        args=(engine_cls, image.copy(), self.lang, cfg,
                            page_results_collector, engine_name)
                    )
                    threads.append(t)
                    t.start()
                    logger.debug(f"Started thread for {engine_name} on page {page_number}")
                except Exception as e:
                    logger.error(f"Could not start OCR for '{engine_name}': {e}")
                    # Immediately record an error placeholder in correct order position later
                    page_results_collector.append({
                        "engine": engine_name,
                        "text": f"[ERROR: Could not start {engine_name} – {e}]"
                    })

            # Wait for all to finish
            for t in threads:
                t.join()
            logger.info(f"All engines finished for Page {page_number}")

            # Sort collector to the same order as self.engine_names
            page_results_collector.sort(
                key=lambda rec: self.engine_names.index(rec["engine"])
                            if rec["engine"] in self.engine_names else -1
            )

            # Extract just the text strings
            page_texts: List[str] = [rec["text"] for rec in page_results_collector]
            all_pages_outputs.append(page_texts)
        # print(f"Num. Pages: {len(all_pages_outputs)}")
        # print(f"Num. Engines: {len(all_pages_outputs[0])}")
        # with open("D:\\ASU\\sem 10\\GRAD PROJ\\EvenBetterOCR\\test\\input\\out.txt", "w", encoding="utf-8") as f:
        #     f.write(all_pages_outputs.__str__())
        logger.info("OCR pipeline completed for all pages.")
        return all_pages_outputs
