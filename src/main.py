import argparse
import json
import logging
import os
from typing import Any, List, Dict, Optional 
import tempfile 
import uuid

from src.llm.clients.gemini_client import GeminiClient 


from .parsers.parser import DocumentParser
from .combiner.combiner import OCRCombiner
from .engines.EngineRegistry import EngineRegistry
from .engines.concrete_implementations.easyOCR import EasyOCREngine # Keep if you might re-add
from .engines.concrete_implementations.suryaOCR import SuryaOCREngine
from .engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from .llm.clients.groq_client import GroqClient
from .llm.llm_processor import LLMProcessor
from PIL import Image # Keep for display logic or if engines need it explicitly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BetterOCR_Core")

AVAILABLE_ENGINES = {
    # "easyocr": EasyOCREngine,
    "suryaocr": SuryaOCREngine,
    "tesseractocr": TesseractOCREngine
}
DEFAULT_DETECTOR_ENGINE = 'suryaocr' 

def setup_global_logging_level(verbose_level: int):
    if verbose_level == 1:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Logging level set to INFO.")
    elif verbose_level >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Logging level set to DEBUG.")
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        logger.warning("Logging level set to WARNING.")

def run_ocr_processing(args_dict: Dict[str, Any]) -> str:
    """
    Core OCR processing logic, taking arguments as a dictionary.
    Returns the final processed text as a string.
    """
    setup_global_logging_level(args_dict.get("verbose", 0))
    logger.info(f"Core processing started with effective arguments: {args_dict}")

    engine_registry = EngineRegistry()
    # User-selected engines will now primarily be for recognition if new pipeline is used
    user_selected_engines = args_dict.get("ocr_engines", [])
    
    for name, cls in AVAILABLE_ENGINES.items():
        engine_registry.register_engine(name, cls)

    doc_parser = DocumentParser()

    try:
        engine_configs_json = args_dict.get("engine_configs_json", "{}")
        engine_configs = json.loads(engine_configs_json)
        if not isinstance(engine_configs, dict):
            logger.warning("engine_configs_json was not a dict. Using empty configs.")
            engine_configs = {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON string for engine_configs_json: {e}. Using empty configs.")
        engine_configs = {}

    document_path = args_dict["document_path"]


    detector_engine_cli = args_dict.get("detector_engine", DEFAULT_DETECTOR_ENGINE)
    if detector_engine_cli not in AVAILABLE_ENGINES:
        logger.error(f"Specified detector engine '{detector_engine_cli}' not available. Using default '{DEFAULT_DETECTOR_ENGINE}'.")
        detector_engine_cli = DEFAULT_DETECTOR_ENGINE
    

    recognizer_engines_cli = user_selected_engines
    if not recognizer_engines_cli: # If user didn't specify --ocr_engines
        recognizer_engines_cli = [name for name in AVAILABLE_ENGINES.keys()] # Use all available
    

    merge_pair_cli = None
    if args_dict.get("use_line_merging", False) and len(recognizer_engines_cli) >= 2:

        primary_recognizer_for_merge = recognizer_engines_cli[0]
        secondary_recognizer_for_merge = recognizer_engines_cli[1]
        merge_pair_cli = (primary_recognizer_for_merge, secondary_recognizer_for_merge)
        logger.info(f"Line merging enabled for pair: {merge_pair_cli}")
    elif args_dict.get("use_line_merging", False):
        logger.warning("Line merging enabled, but fewer than two recognizer engines specified. Merging will be skipped.")

    merger_config_cli = args_dict.get("line_merger_config", {}) # Expects dict, not JSON string here
    if isinstance(merger_config_cli, str): # Handle if it was accidentally passed as JSON string
        try: merger_config_cli = json.loads(merger_config_cli)
        except: merger_config_cli = {}


    combiner = OCRCombiner(
        engine_registry=engine_registry,
        document_parser=doc_parser,
        document_path=document_path,
        lang_list=args_dict.get("lang", ["ar"]),
        detector_engine_name=detector_engine_cli,
        recognizer_engine_names=recognizer_engines_cli,
        engine_configs=engine_configs,
        merge_recognizer_outputs_pair=merge_pair_cli, 
        line_merger_config=merger_config_cli
    )


    logger.info("Starting OCR processing with detection, parallel recognition, and optional merging...")
    final_structured_document = combiner.run_detection_then_parallel_recognition_and_merge()
    
    page_level_ocr_content_strings: List[str] = []
    page_average_confidences: List[float] = []

    if final_structured_document:
        for page_idx, page_data_lines in enumerate(final_structured_document):
            page_text = OCRCombiner.reassemble_text_from_structured(page_data_lines)
            page_level_ocr_content_strings.append(page_text)
            
            line_confs = [ld.get('text_confidence', 0.0) 
                          for ld in page_data_lines 
                          if isinstance(ld.get('text_confidence'), (float, int))]
            avg_page_conf = sum(line_confs) / len(line_confs) if line_confs else 0.0
            page_average_confidences.append(avg_page_conf)

            if args_dict.get("verbose", 0) >= 1: # INFO level
                logger.info(f"Page {page_idx+1} Processed Text (AvgConf: {avg_page_conf:.2f}): {page_text[:150].replace(os.linesep, ' ')}...")
    else:
        logger.warning("The OCR pipeline returned no structured document data.")


    
    final_processed_content_list: List[str] = []
    llm_refinement_threshold = args_dict.get("llm_refinement_threshold", 0.75) # TODO: Adjust threshold

    if args_dict.get("use_llm", True): 
        logger.info("LLM refinement is enabled.")
        if not args_dict.get("groq_api_key") and not os.environ.get("GROQ_API_KEY"):
            logger.warning("Groq API key not provided. LLM refinement will be skipped.")
            final_processed_content_list = page_level_ocr_content_strings
        elif not page_level_ocr_content_strings:
            logger.warning("No OCR content strings to refine with LLM.")
            final_processed_content_list = []
        else:
            try:
                
                # Use gemini 2.0 instead of groq
                # groq_client = GroqClient(model_name=args_dict.get("llm_model_name", "gemma2-9b-it"), api_token=groq_api_key)
                llm_client = GeminiClient()
                llm_processor = LLMProcessor(llm_client=llm_client)
                llm_model_kwargs = {"temperature": args_dict.get("llm_temp", 0.0)}
                lang_list_str_for_llm = ", ".join(args_dict.get("lang", ["ar"]))
                
                logger.info(f"Starting LLM refinement for {len(page_level_ocr_content_strings)} pages if confidence is below {llm_refinement_threshold}...")

                for i, single_page_text_to_refine in enumerate(page_level_ocr_content_strings):
                    page_num = i + 1
                    current_page_avg_conf = page_average_confidences[i] if i < len(page_average_confidences) else 0.0

                    if current_page_avg_conf < llm_refinement_threshold:
                        logger.info(f"Refining Page {page_num} with LLM (confidence {current_page_avg_conf:.2f} < {llm_refinement_threshold}).")
                        # LLMProcessor.refine_text expects List[str]. We give it a list with one item.
                        raw_llm_response = llm_processor.refine_text(
                            [single_page_text_to_refine], # Pass as a list with one string
                            lang_list_str=lang_list_str_for_llm,
                            context_keywords=args_dict.get("llm_context_keywords", ""),
                            model_kwargs=llm_model_kwargs
                        )
                        if "[LLM_ERROR" not in raw_llm_response.upper():
                            # refined_page_text = llm_processor.parse_llm_output(raw_llm_response)
                            refined_page_text = raw_llm_response
                            if "[LLM_PARSE_ERROR" in refined_page_text.upper():
                                logger.error(f"LLM parsing failed for Page {page_num}. Using raw LLM response. Details: {refined_page_text}")
                                final_processed_content_list.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_PARSE_ERROR: {raw_llm_response}]")
                            else:
                                final_processed_content_list.append(refined_page_text)
                                logger.info(f"LLM refinement complete for Page {page_num}.")
                        else:
                            logger.error(f"LLM refinement API call failed for Page {page_num}. LLM response: {raw_llm_response}")
                            final_processed_content_list.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_API_ERROR: {raw_llm_response}]")
                    else: # Confidence is good, no LLM refinement for this page
                        logger.info(f"Skipping LLM refinement for Page {page_num} (confidence {current_page_avg_conf:.2f} >= {llm_refinement_threshold}).")
                        final_processed_content_list.append(single_page_text_to_refine) # Use the merged/recognized text
                logger.info("LLM refinement process completed for applicable pages.")

            except Exception as e:
                logger.error(f"Critical error during LLM processing: {e}. Output will be pre-LLM.", exc_info=True)
                final_processed_content_list = page_level_ocr_content_strings # Fallback
    else: # LLM not used
        logger.info("LLM refinement is disabled.")
        final_processed_content_list = page_level_ocr_content_strings


    
    output_string_for_file = ""
    if final_processed_content_list:
        page_texts_for_output = []
        for i, page_text_content in enumerate(final_processed_content_list):
            header = f"\n--- Page {i+1} (Confidence: {page_average_confidences[i]:.2f} " \
                     f"{'after LLM' if args_dict.get('use_llm', True) and page_average_confidences[i] < llm_refinement_threshold else 'pre-LLM or LLM skipped'}) --- \n"
            page_texts_for_output.append(header + page_text_content)
        output_string_for_file = "\n".join(page_texts_for_output)
    elif not final_processed_content_list and not page_level_ocr_content_strings:
        output_string_for_file = "[No content processed or error in initial stages]"
    else:
        output_string_for_file = "[No text content extracted after processing]"
        
    logger.info("Core OCR processing finished.")
    print(output_string_for_file)
    return final_processed_content_list


def main_cli():
    parser = argparse.ArgumentParser(description="EvenBetterOCR: Advanced OCR Processing Pipeline.")
    parser.add_argument("document_path", type=str, help="Path to the document (PDF or image file) to process.")
    # --ocr_engines now refers to RECOGNIZER engines for the new pipeline
    parser.add_argument("--ocr_engines", nargs="+", type=str, default=['suryaocr', 'tesseractocr'], # Sensible default recognizers
                        choices=list(AVAILABLE_ENGINES.keys()), help="List of OCR engines to use for RECOGNITION.")
    parser.add_argument("--detector_engine", type=str, default=DEFAULT_DETECTOR_ENGINE,
                        choices=list(AVAILABLE_ENGINES.keys()), help="OCR engine to use for initial layout/text line DETECTION.")
    
    parser.add_argument("--lang", nargs="+", type=str, default=["ar"], help="List of language codes for OCR.")
    parser.add_argument("--engine_configs_json", type=str, default="{}", help="JSON string with specific configurations for engines.")

    parser.add_argument("--use_line_merging", action=argparse.BooleanOptionalAction, default=True, # Default to True for new pipeline
                        help="Enable/Disable line-level merging of specified OCR recognizer outputs.")
    parser.add_argument("--line_merger_config_json", type=str, default='{"min_wordfreq_for_dict_check": 1e-7}', # Example for LinePairMerger
                        help="JSON string for LinePairMerger configuration.")


    parser.add_argument("--use_llm", action=argparse.BooleanOptionalAction, default=True, help="Enable/Disable LLM refinement.")
    parser.add_argument("--llm_refinement_threshold", type=float, default=0.80, # Example: refine if page confidence < 80%
                        help="Average page confidence threshold below which LLM refinement is triggered.")
    parser.add_argument("--llm_model_name", type=str, default="gemma2-9b-it", help="Groq LLM model name.")
    parser.add_argument("--groq_api_key", type=str, default=os.environ.get("GROQ_API_KEY"), help="Groq API key.")
    parser.add_argument("--llm_context_keywords", type=str, default="", help="Optional context keywords for LLM.")
    parser.add_argument("--llm_temp", type=float, default=0.0, help="Temperature for LLM generation.")

    parser.add_argument("--display_bounding_boxes", type=str, choices=list(AVAILABLE_ENGINES.keys()) + [None], default=None,
                        help="Display bounding boxes from a specific engine for the first page (GUI).")
    parser.add_argument("--display_annotated_output", type=str, choices=list(AVAILABLE_ENGINES.keys()) + [None], default=None,
                        help="Display annotated output from a specific engine for the first page (GUI).")
    # New display options for the new pipeline
    parser.add_argument("--display_layout_regions", action="store_true", help="Display detected layout regions (uses detector_engine).")
    parser.add_argument("--display_detected_lines", action="store_true", help="Display detected text lines after layout filtering (uses detector_engine).")


    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity (-v INFO, -vv DEBUG).")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save final text output.")

    args = parser.parse_args()
    args_dict = vars(args)

    # Parse line_merger_config_json into a dict
    try:
        args_dict["line_merger_config"] = json.loads(args.line_merger_config_json)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON for --line_merger_config_json. Using empty dict. Value: {args.line_merger_config_json}")
        args_dict["line_merger_config"] = {}


    if not os.environ.get("FLASK_RUNNING"): 
        doc_parser_display = DocumentParser()
        images_for_display = None
        try:
            if args.document_path: # Ensure document_path is available
                 images_for_display = doc_parser_display.load_images_from_document(args.document_path)
        except Exception as e:
            logger.error(f"Failed to load document for display: {e}")

        if images_for_display and images_for_display[0]:
            first_page_image_disp = images_for_display[0].copy()
            
            # Instantiate engine_configs for display part as well
            engine_configs_disp_main = {}
            try: engine_configs_disp_main = json.loads(args.engine_configs_json)
            except: pass

            if args.display_layout_regions or args.display_detected_lines:
                detector_disp_name = args.detector_engine
                detector_cls_disp = AVAILABLE_ENGINES.get(detector_disp_name)
                if detector_cls_disp:
                    detector_config_disp = engine_configs_disp_main.get(detector_disp_name, {})
                    # Ensure use_recognizer is False if Surya is just detecting for display
                    if detector_disp_name == 'suryaocr' and 'use_recognizer' not in detector_config_disp:
                        detector_config_disp['use_recognizer'] = False
                    try:
                        detector_inst_disp = detector_cls_disp(lang_list=args.lang, **detector_config_disp)
                        if args.display_layout_regions and hasattr(detector_inst_disp, 'display_layout_regions'):
                            logger.info(f"Displaying layout regions using {detector_disp_name}...")
                            detector_inst_disp.display_layout_regions(first_page_image_disp)
                        if args.display_detected_lines and hasattr(detector_inst_disp, 'display_detected_text_lines'):
                            logger.info(f"Displaying detected text lines (layout filtered) using {detector_disp_name}...")
                            detector_inst_disp.display_detected_text_lines(first_page_image_disp, with_layout_filtering=True)
                    except Exception as e:
                        logger.error(f"Error during advanced display for {detector_disp_name}: {e}", exc_info=True)

            display_engine_name_legacy = args.display_bounding_boxes or args.display_annotated_output
            if display_engine_name_legacy:
                engine_cls_legacy_disp = AVAILABLE_ENGINES.get(display_engine_name_legacy)
                if engine_cls_legacy_disp:
                    legacy_engine_config_disp = engine_configs_disp_main.get(display_engine_name_legacy, {})
                    try:
                        engine_legacy_disp = engine_cls_legacy_disp(lang_list=args.lang, **legacy_engine_config_disp)

                        if args.display_bounding_boxes:
                            logger.info(f"Displaying legacy bounding boxes for {display_engine_name_legacy}...")
                            engine_legacy_disp.display_bounding_boxes(first_page_image_disp)
                        if args.display_annotated_output:
                            logger.info(f"Displaying legacy annotated output for {display_engine_name_legacy}...")
                            engine_legacy_disp.display_annotated_output(first_page_image_disp)
                    except Exception as e:
                         logger.error(f"Error during legacy display for {display_engine_name_legacy}: {e}", exc_info=True)


    final_text_output = run_ocr_processing(args_dict)

    print("\n--- Final Processed Text (from main_cli) ---")
    print(final_text_output) # This now includes page numbers and confidence
    # print("--- End of Final Processed Text ---") # Redundant with header in string

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(final_text_output)
            logger.info(f"Final output saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write output to file {args.output_file}: {e}", exc_info=True)

    logger.info("BetterOCR CLI processing finished.")

if __name__ == "__main__":
    main_cli()