
import threading
import logging
from typing import Any, List, Dict, Tuple, Type
from PIL import Image

from ..engines.IEngine import OCREngine
from ..engines.EngineRegistry import EngineRegistry # To get engine classes
from ..parsers.parser import DocumentParser

from .wordMerger import WordMerger

logger = logging.getLogger(__name__)

class OCRCombiner:
    def __init__(self, 
                 engine_registry: EngineRegistry,
                 engine_names: List[str],
                 document_parser: DocumentParser,
                 document_path: str,
                 lang_list: List[str], 
                 engine_configs: Dict[str, Dict] = None,
                 merge_engine_outputs: List[Tuple[str, str]] = None, # e.g., [('suryaocr', 'tesseractocr')]
                 word_merger_config: Dict[str, Any] = None): # Config for WordMerger
        """
        Initializes the OCRCombiner.
        Args:
            engine_registry:        Instance of EngineRegistry to fetch engine classes.
            engine_names:           List of names of OCR engines to use (e.g., ["easyocr", "tesseractocr"]).
            document_parser:        Instance of DocumentParser.
            document_path:          Path to the document to be processed.
            lang:                   List of language codes for OCR engines.
            engine_configs:         Optional dictionary with specific configurations for each engine.
                                    Format: {"engine_name": {"param1": "value1", ...}}
            merge_engine_outputs:   Which engines to merge together e.g., [('suryaocr', 'tesseractocr')]
            word_merger_config:     Config for WordMerger
        """
        self.engine_registry = engine_registry
        self.engine_names = engine_names
        self.document_parser = document_parser
        self.document_path = document_path
        self.lang_list = lang_list
        self.engine_configs = engine_configs if engine_configs else {}
        
        self.merge_engine_outputs = merge_engine_outputs if merge_engine_outputs else []
        self.word_merger = None
        if self.merge_engine_outputs:
            merger_cfg = word_merger_config if word_merger_config else {}
            self.word_merger = WordMerger(**merger_cfg)
            logger.info(f"WordMerger initialized to merge pairs: {self.merge_engine_outputs}")

        logger.info(f"OCRCombiner initialized for engines: {engine_names} on document: {document_path}")

########################################################################################### WORD MERGING PART ###########################################################################################
    def run_ocr_pipeline_parallel_structured(self) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Processes the document using multiple OCR engines in parallel, fetching STRUCTURED output.
        Returns:
            A dictionary mapping engine_name to its list of structured outputs per page.
            {
                "suryaocr": [ [{'bbox':..., 'text':..., 'confidence':...}, ...], # Page 1 for Surya
                              [{'bbox':..., 'text':..., 'confidence':...}, ...]  # Page 2 for Surya
                            ],
                "tesseractocr": [ [...], [...] ]
            }
        """
        logger.info(f"Starting STRUCTURED OCR pipeline for document: {self.document_path}")
        try:
            all_document_images = self.document_parser.load_images_from_document(self.document_path)
        except Exception as e:
            logger.error(f"Failed to load document pages: {e}")
            # Return a structure indicating error for all requested engines
            error_page_struct = [{'bbox': [], 'text': f"[ERROR: Failed to load document – {e}]", 'confidence': 0}]
            return {name: [[error_page_struct]] for name in self.engine_names}


        if not all_document_images:
            logger.warning("No images found in document.")
            error_page_struct = [{'bbox': [], 'text': "[ERROR: No images to process]", 'confidence': 0}]
            return {name: [[error_page_struct]] for name in self.engine_names}

        num_pages = len(all_document_images)
        logger.info(f"Document loaded with {num_pages} pages.")

        # engine_name -> List_of_pages, where each page is List_of_word_dicts
        engine_structured_results_map: Dict[str, List[List[Dict[str, Any]]]] = {name: [] for name in self.engine_names}
        threads: List[threading.Thread] = []


        # Corrected threading logic:
        # Each thread will compute all pages for ONE engine and store it in engine_structured_results_map[engine_name]
        def thread_target_per_engine(engine_cls, images, lang_list, cfg, engine_name_key, output_map):
            try:
                engine_instance = engine_cls(lang_list=lang_list, **cfg)
                logger.info(f"[{engine_name_key}] Starting OCR for structured output on {len(images)} image(s)…")
                structured_outputs_all_pages = engine_instance.get_structured_output(images)
                output_map[engine_name_key] = structured_outputs_all_pages
                logger.info(f"[{engine_name_key}] Finished structured OCR; {len(structured_outputs_all_pages)} pages processed.")
            except Exception as e:
                logger.error(f"[{engine_name_key}] Error during structured OCR: {e}", exc_info=True)
                error_output = [{'bbox': [], 'text': f"[ERROR: {engine_name_key} failed – {e}]", 'confidence': 0}]
                output_map[engine_name_key] = [[error_output] for _ in range(len(images))]


        for engine_name in self.engine_names:
            try:
                engine_cls = self.engine_registry.get_engine_class(engine_name)
                cfg = self.engine_configs.get(engine_name, {})

                t = threading.Thread(
                    target=thread_target_per_engine,
                    args=(engine_cls, all_document_images, self.lang_list, cfg,
                          engine_name, engine_structured_results_map)
                )
                threads.append(t)
                t.start()
                logger.debug(f"Started thread for {engine_name} (structured) to process all {num_pages} pages.")
            except Exception as e:
                logger.error(f"Could not start structured OCR thread for '{engine_name}': {e}", exc_info=True)
                error_output = [{'bbox': [], 'text': f"[ERROR: Could not start thread for {engine_name} – {e}]", 'confidence': 0}]
                engine_structured_results_map[engine_name] = [[error_output] for _ in range(num_pages)]

        for t in threads:
            t.join()
        logger.info(f"All engine threads (structured) finished processing all pages.")
        return engine_structured_results_map


    def run_ocr_and_merge(self) -> List[List[Dict[str, Any]]]: # Returns list of pages, each page is a list of merged word dicts
        """
        Runs the OCR pipeline, fetches structured outputs, and merges specified pairs.
        Falls back to a primary engine if merging isn't specified or possible for a pair.
        """
        # 1. Get structured output from all engines
        #    This returns: {'suryaocr': [page1_struct, page2_struct,...], 'tesseractocr': [page1_struct, ...]}
        all_engine_structured_outputs = self.run_ocr_pipeline_parallel_structured()

        num_pages = 0
        if self.engine_names and all_engine_structured_outputs.get(self.engine_names[0]):
            num_pages = len(all_engine_structured_outputs[self.engine_names[0]])
        
        if num_pages == 0:
            logger.warning("No pages found or processed, cannot merge.")
            return []

        final_merged_pages: List[List[Dict[str, Any]]] = []

        # For now, let's assume one merge pair, e.g., ('suryaocr', 'tesseractocr')
        # suryaocr is the primary to fallback to.
        primary_engine_for_fallback = 'suryaocr' # Configurable later
        if primary_engine_for_fallback not in self.engine_names:
            primary_engine_for_fallback = self.engine_names[0] if self.engine_names else None


        for page_idx in range(num_pages):
            page_merged_output = None
            merged_this_page = False

            if self.word_merger and self.merge_engine_outputs:
                # Assuming the first pair in merge_engine_outputs for now
                engine1_name, engine2_name = self.merge_engine_outputs[0]

                if engine1_name in all_engine_structured_outputs and engine2_name in all_engine_structured_outputs:
                    items1 = all_engine_structured_outputs[engine1_name][page_idx]
                    items2 = all_engine_structured_outputs[engine2_name][page_idx]
                    
                    # Ensure items are not error strings before passing to merger
                    valid_items1 = isinstance(items1, list) and all(isinstance(item, dict) for item in items1)
                    valid_items2 = isinstance(items2, list) and all(isinstance(item, dict) for item in items2)

                    if not valid_items1:
                        logger.warning(f"Page {page_idx+1}: Invalid structured data for {engine1_name}, using fallback or empty.")
                        items1 = []
                    if not valid_items2:
                        logger.warning(f"Page {page_idx+1}: Invalid structured data for {engine2_name}, using fallback or empty.")
                        items2 = []
                    page_merged_output = self.word_merger.merge_page_outputs(items1, engine1_name, items2, engine2_name)
                    merged_this_page = True

                else:
                    logger.warning(f"Page {page_idx+1}: One or both engines for merging ({engine1_name}, {engine2_name}) not found in results. Falling back.")
            
            if not merged_this_page:
                if primary_engine_for_fallback and primary_engine_for_fallback in all_engine_structured_outputs:
                    logger.info(f"Page {page_idx+1}: Using fallback engine '{primary_engine_for_fallback}'.")
                    fallback_data = all_engine_structured_outputs[primary_engine_for_fallback][page_idx]
                    if isinstance(fallback_data, list) and all(isinstance(item, dict) for item in fallback_data):
                         page_merged_output = fallback_data
                    else:
                         page_merged_output = [{'bbox': [], 'text': f"[ERROR: Invalid fallback data for {primary_engine_for_fallback}]", 'confidence': 0}]
                else:
                    logger.warning(f"Page {page_idx+1}: No merge happened and no primary fallback engine available or its data is missing.")
                    page_merged_output = [{'bbox': [], 'text': "[ERROR: No data after merge/fallback attempts]", 'confidence': 0}]
            
            final_merged_pages.append(page_merged_output)

        logger.info("Structured OCR and merging pipeline completed.")
        return final_merged_pages

    @staticmethod
    def reassemble_text_from_structured(structured_page_output: List[Dict[str, Any]], line_break_char='\n', word_separator=' ') -> str:
        """
        Reassembles a full text string from a list of word/block dictionaries.
        This is a basic implementation; more sophisticated line/paragraph reconstruction might be needed.
        Assumes items are somewhat sorted (e.g., top-to-bottom, left-to-right).
        """
        if not structured_page_output or not isinstance(structured_page_output[0], dict):
            # Handle cases where it might be an error string or improperly formatted
            if isinstance(structured_page_output, list) and structured_page_output and isinstance(structured_page_output[0], str): # list of strings
                return structured_page_output[0] # if it's an error string in a list
            if isinstance(structured_page_output, str): # if it's just an error string
                return structured_page_output
            return ""

        # Basic approach: join all texts. A better way would be to sort by y then x, and infer lines.
        # For simplicity now, just join.
        # This will be used if LLM needs a single string input after merging.
        texts = [item.get('text', '') for item in structured_page_output if isinstance(item, dict) and item.get('text')]
        return word_separator.join(texts) # This doesn't preserve lines well.
########################################################################################### WORD MERGING PART ###########################################################################################

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
            results_dict:    A thread-safe dictionary to which we append:
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
        # all_pages_outputs = [ [page_1_tess, page_2_surya], [page_1_tess, page_2_surya], ]

        return all_pages_outputs