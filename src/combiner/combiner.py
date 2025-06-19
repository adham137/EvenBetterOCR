# src/combiner/combiner.py

import threading
import logging
from typing import Any, List, Dict, Tuple, Type, Optional
from PIL import Image

# Assuming correct relative imports for your project structure
from ..engines.IEngine import OCREngine
from ..engines.EngineRegistry import EngineRegistry
from ..parsers.parser import DocumentParser
# Ensure this is your latest merger class (LineROVERMerger or LineROVERMerger)
from .lineMerger import LineROVERMerger # Or your chosen name (e.g. LineROVERMerger)

# Import specific engine classes if needed for type hinting or direct instantiation (though registry is preferred)
from ..engines.concrete_implementations.suryaOCR import SuryaOCREngine
from ..engines.concrete_implementations.tesseractOCR import TesseractOCREngine


logger = logging.getLogger(__name__)

class OCRCombiner:
    def __init__(self,
                 engine_registry: EngineRegistry,
                 document_parser: DocumentParser,
                 document_path: str,
                 lang_list: List[str],
                 # Engine used for initial layout/text line detection
                 detector_engine_name: str = 'suryaocr', # Default to Surya for detection
                 # Engines used for recognition on detected lines
                 recognizer_engine_names: Optional[List[str]] = None,
                 engine_configs: Dict[str, Dict] = None,
                 # For merging recognizer outputs
                 merge_recognizer_outputs_pair: Optional[Tuple[str, str]] = None, # e.g., ('suryaocr', 'tesseractocr')
                 line_merger_config: Dict[str, Any] = None): # Config for LineROVERMerger

        self.engine_registry = engine_registry
        self.document_parser = document_parser
        self.document_path = document_path
        self.lang_list = lang_list
        self.engine_configs = engine_configs if engine_configs else {}

        self.detector_engine_name = detector_engine_name
        # If recognizer_engine_names are not provided, assume all registered engines (excluding detector if same)
        # or make it a requirement. For now, let's assume they are distinct or handled.
        self.recognizer_engine_names = recognizer_engine_names if recognizer_engine_names else []

        self.merge_recognizer_outputs_pair = merge_recognizer_outputs_pair
        self.line_merger: Optional[LineROVERMerger] = None # Type hint for clarity
        if self.merge_recognizer_outputs_pair and len(self.recognizer_engine_names) >= 2:
            merger_cfg = line_merger_config if line_merger_config else {}
            # Ensure LineROVERMerger (or your chosen class name) is correctly imported
            self.line_merger = LineROVERMerger(lang=self.lang_list[0] if self.lang_list else "ar", **merger_cfg) # Pass lang
            logger.info(f"LineROVERMerger initialized to merge pair: {self.merge_recognizer_outputs_pair} with config: {merger_cfg}")
        elif self.merge_recognizer_outputs_pair:
             logger.warning("Merge pair specified, but not enough recognizer engines to merge.")


        logger.info(f"OCRCombiner initialized. Detector: {self.detector_engine_name}, Recognizers: {self.recognizer_engine_names}, Document: {document_path}")


    def _run_recognition_on_detected_lines(
        self,
        engine_name: str, # Name of the recognizer engine
        full_page_images: List[Image.Image], # List of original full page images
        all_pages_detected_lines_info: List[List[Dict[str, Any]]],
        output_recognition_results_map: Dict[str, List[List[Dict[str, Any]]]]
    ):
        """
        Target function for a thread. Runs one OCR RECOGNITION engine on pre-detected lines.
        """
        try:
            recognizer_engine_class = self.engine_registry.get_engine_class(engine_name)
            recognizer_config = self.engine_configs.get(engine_name, {})


            recognizer_instance = recognizer_engine_class(lang_list=self.lang_list, **recognizer_config)
            logger.info(f"[{engine_name}] Starting RECOGNITION on detected lines for {len(full_page_images)} pages.")

            #TODO: Unify the calling function for the OCR engines

            # Check if the engine has a dedicated 'recognize_detected_lines' or similar method
            if engine_name == 'tesseractocr' and hasattr(recognizer_instance, 'recognize_detected_lines') and callable(getattr(recognizer_instance, 'recognize_detected_lines')):
                # This is the preferred method for TesseractOCREngine
                recognized_pages_data = recognizer_instance.recognize_detected_lines(
                    full_page_images,
                    all_pages_detected_lines_info
                )
            elif engine_name == 'suryaocr' and hasattr(recognizer_instance, 'get_structured_output') and callable(getattr(recognizer_instance, 'get_structured_output')):
                # This is the preferred method for SuryaOCREngine
                recognized_pages_data = recognizer_instance.get_structured_output(
                    full_page_images,
                    input_detections=all_pages_detected_lines_info # Pass detected lines
                )
            else:
                logger.error(f"Engine {engine_name} does not have a suitable method for recognizing pre-detected lines.")
                raise NotImplementedError(f"{engine_name} cannot recognize pre-detected lines.")

            output_recognition_results_map[engine_name] = recognized_pages_data
            logger.info(f"[{engine_name}] Finished RECOGNITION; {len(recognized_pages_data)} pages processed.")

        except Exception as e:
            logger.error(f"[{engine_name}] Error during RECOGNITION: {e}", exc_info=True)
            # Populate with error placeholders for all pages, preserving structure
            error_output_for_page = [{
                'bbox': line_info.get('bbox'), 'label': line_info.get('label', '[ERROR]'),
                'text': f"[ERROR_RECOGNITION: {engine_name} failed – {e}]",
                'text_confidence': 0.0
            } for line_info in (all_pages_detected_lines_info[0] if all_pages_detected_lines_info else [{'bbox':[]}])] # Use first page structure as template
            
            output_recognition_results_map[engine_name] = [error_output_for_page for _ in all_pages_detected_lines_info]


    def run_detection_then_parallel_recognition_and_merge(self) -> List[List[Dict[str, Any]]]:
        """
        Main pipeline:
        1. Load images.
        2. Run initial detection (layout + text lines) using the detector_engine.
        3. Run specified recognizer_engines IN PARALLEL on these detected lines.
        4. Merge outputs of two specified recognizer engines.
        5. Fallback if merging not possible or not enough engines.
        """
        logger.info("Starting New Pipeline: Detection -> Parallel Recognition -> Merge")

        # 1. Load Images
        try:
            all_document_images = self.document_parser.load_images_from_document(self.document_path)
        except Exception as e:
            logger.error(f"Failed to load document pages: {e}")
            return [[{'text': f"[ERROR: Failed to load document – {e}]", 'confidence': 0}]] # Single page error
        if not all_document_images:
            logger.warning("No images found in document.")
            return []
        num_pages = len(all_document_images)
        logger.info(f"Document loaded with {num_pages} pages.")

        # 2. Initial Detection (using detector_engine, e.g., SuryaOCREngine)
        try:
            detector_class = self.engine_registry.get_engine_class(self.detector_engine_name)
            detector_config = self.engine_configs.get(self.detector_engine_name, {})
            if self.detector_engine_name == 'suryaocr' and 'use_recognizer' not in detector_config:
                 detector_config['use_recognizer'] = False
            
            detector_instance = detector_class(lang_list=self.lang_list, **detector_config)

            if not hasattr(detector_instance, 'detect_text_lines_with_layout'):
                logger.error(f"Detector engine {self.detector_engine_name} does not have 'detect_text_lines_with_layout' method.")
                raise NotImplementedError("Detector engine misconfigured for line detection.")
            
            logger.info(f"Running detection using {self.detector_engine_name}...")
            all_pages_detected_lines_info = detector_instance.detect_text_lines_with_layout(all_document_images)
            logger.info(f"Detection complete. Found lines for {len(all_pages_detected_lines_info)} pages.")

        except Exception as e:
            logger.error(f"Error during detection phase with {self.detector_engine_name}: {e}", exc_info=True)
            return [[{'text': f"[ERROR: Detection failed - {e}]", 'confidence': 0} for _ in range(num_pages)]]


        # 3. Parallel Recognition
        # RecognizedLineDict: {'bbox':..., 'label':..., 'text':..., 'text_confidence':..., 'words':...}
        recognition_results_map: Dict[str, List[List[Dict[str, Any]]]] = {}
        threads: List[threading.Thread] = []

        if not self.recognizer_engine_names:
            logger.warning("No recognizer engines specified. Returning detected lines without text recognition.")
            # We could return all_pages_detected_lines_info here, but the expected output is recognized lines.
            # For now, return empty or error.
            return [[{'text': "[NO_RECOGNIZERS_CONFIGURED]", 'confidence':0.0}] for _ in range(num_pages)]


        for rec_engine_name in self.recognizer_engine_names:
            if not all_pages_detected_lines_info and num_pages > 0: # If detection yielded nothing for any page
                 logger.warning(f"No lines detected by {self.detector_engine_name}, cannot run recognizer {rec_engine_name}.")
                 recognition_results_map[rec_engine_name] = [[] for _ in range(num_pages)] # Empty results for this engine
                 continue

            try:
                t = threading.Thread(
                    target=self._run_recognition_on_detected_lines,
                    args=(rec_engine_name, all_document_images, all_pages_detected_lines_info,
                          recognition_results_map)
                )
                threads.append(t)
                t.start()
                logger.debug(f"Started RECOGNITION thread for {rec_engine_name}.")
            except Exception as e:
                logger.error(f"Could not start RECOGNITION thread for '{rec_engine_name}': {e}", exc_info=True)
                recognition_results_map[rec_engine_name] = [[{'text':f"[ERROR_THREAD_START_RECO:{e}]"}] for _ in range(num_pages)]


        for t in threads:
            t.join()
        logger.info(f"All RECOGNITION engine threads finished.")

        # 4. Merge Recognizer Outputs (if configured)
        final_document_output: List[List[Dict[str, Any]]]

        if self.line_merger and self.merge_recognizer_outputs_pair and \
           self.merge_recognizer_outputs_pair[0] in recognition_results_map and \
           self.merge_recognizer_outputs_pair[1] in recognition_results_map:
            
            engine_A_name, engine_B_name = self.merge_recognizer_outputs_pair
            logger.info(f"Merging outputs from {engine_A_name} and {engine_B_name}...")
            
            doc_results_A = recognition_results_map[engine_A_name]
            doc_results_B = recognition_results_map[engine_B_name]

            # Ensure both results have the same number of pages as detected_lines_info
            if len(doc_results_A) != num_pages or len(doc_results_B) != num_pages:
                logger.error(f"Page count mismatch after recognition. A:{len(doc_results_A)}, B:{len(doc_results_B)}, Expected:{num_pages}. Cannot merge reliably.")
                # Fallback: Pick one engine's results or error.
                fallback_rec_name = self.recognizer_engine_names[0]
                final_document_output = recognition_results_map.get(fallback_rec_name, [[{'text': "[MERGE_PAGE_COUNT_ERROR]"}] for _ in range(num_pages)])
            else:
                final_document_output = self.line_merger.merge_document_results(
                    doc_results_A, doc_results_B, engine_A_name, engine_B_name
                )
                logger.info("Merging complete.")
        else:
            # Fallback: No merger, or specified engines for merging not available/recognized.
            # Use results from the first available recognizer, or just the detected lines structure if no recognizers ran.
            if self.recognizer_engine_names:
                primary_recognizer_name = self.recognizer_engine_names[0]
                if primary_recognizer_name in recognition_results_map:
                    logger.info(f"No merge performed or merge pair not fully available. Using results from primary recognizer: {primary_recognizer_name}")
                    final_document_output = recognition_results_map[primary_recognizer_name]
                else:
                    logger.warning(f"Primary recognizer {primary_recognizer_name} results not found. Returning empty or detected lines.")
                    # Return structure of detected lines but with error text for recognition
                    final_document_output = [[{**line_info, 'text':'[PRIMARY_RECO_FAILED]', 'text_confidence':0.0} for line_info in page_lines] for page_lines in all_pages_detected_lines_info] if all_pages_detected_lines_info else [[] for _ in range(num_pages)]
            else: # No recognizers ran at all
                logger.warning("No recognizers ran. Output will be based on detected lines structure without text.")
                final_document_output = [[{**line_info, 'text':'[NO_RECOGNITION]', 'text_confidence':0.0} for line_info in page_lines] for page_lines in all_pages_detected_lines_info] if all_pages_detected_lines_info else [[] for _ in range(num_pages)]
        
        return final_document_output


    @staticmethod
    def reassemble_text_from_structured(structured_page_output: List[Dict[str, Any]],
                                         line_break_char='\n', word_separator=' ') -> str:
        # This method now takes a list of recognized line dicts for a single page
        if not structured_page_output: return ""
        
        # Lines are assumed to be somewhat in reading order due to `position` sorting earlier.
        # Each dict in structured_page_output is a line.
        page_text = ""
        for i, line_item in enumerate(structured_page_output):
            if isinstance(line_item, dict) and line_item.get('text'):
                page_text += line_item['text']
                if i < len(structured_page_output) - 1: # Add newline if not the last line
                    page_text += line_break_char
            elif isinstance(line_item, str): # Handle error strings directly
                page_text += line_item + line_break_char

        return page_text.strip()