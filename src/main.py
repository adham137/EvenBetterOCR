import argparse
import json
import logging
import os
from typing import Any, List, Dict, Optional # Added Optional
import tempfile # For handling temporary files if needed by Flask part later
import uuid # For unique temporary file names

from .parsers.parser import DocumentParser
from .combiner.combiner import OCRCombiner # Assuming combiner.py is in the same directory or PYTHONPATH
from .engines.EngineRegistry import EngineRegistry
from .engines.concrete_implementations.easyOCR import EasyOCREngine # Keep if you might re-add
from .engines.concrete_implementations.suryaOCR import SuryaOCREngine
from .engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from .llm.clients.groq_client import GroqClient
from .llm.llm_processor import LLMProcessor

from PIL import Image


# --- Logger Setup --- (Keep as is)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EvenBetterOCR_Core") # Renamed for clarity if main.py is also run as script

AVAILABLE_ENGINES = {
    # "easyocr": EasyOCREngine, # Keep commented if not default
    "suryaocr": SuryaOCREngine,
    "tesseractocr": TesseractOCREngine
}

def setup_global_logging_level(verbose_level: int):
    # (Keep as is)
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

def run_ocr_processing(args_dict: Dict[str, Any]) -> List[str]:
    """
    Core OCR processing logic, taking arguments as a dictionary.
    Returns the final processed text as a string.
    """
    setup_global_logging_level(args_dict.get("verbose", 0))
    logger.info(f"Core processing started with effective arguments: {args_dict}")

    engine_registry = EngineRegistry()
    # Ensure AVAILABLE_ENGINES is accessible or passed if this function moves file
    for name, cls in AVAILABLE_ENGINES.items():
        if name in args_dict.get("ocr_engines", []):
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

    # GUI display options are generally not applicable in server context,
    # but we'll acknowledge them if present in args_dict.
    if args_dict.get("display_bounding_boxes") or args_dict.get("display_annotated_output"):
        logger.info("Display options were provided but are ignored in non-interactive mode.")
        # In a server, you wouldn't typically pop up GUI windows.
        # If you wanted to return images with bboxes, that'd be a different feature.

    document_path = args_dict["document_path"] # This must be provided

    merge_pairs = []
    use_word_merging = args_dict.get("use_word_merging", False) # Get the new flag

    if use_word_merging:
        # Define pairs for merging - this could also be made configurable
        if 'suryaocr' in args_dict.get("ocr_engines", []) and 'tesseractocr' in args_dict.get("ocr_engines", []):
            merge_pairs.append(('suryaocr', 'tesseractocr'))
        logger.info(f"Word merging enabled for pairs: {merge_pairs}")
    else:
        logger.info("Word merging is disabled.")

    # Default merger config, can be overridden if passed in args_dict later
    merger_config = args_dict.get("word_merger_config", {
        "iou_threshold": 0.4,
        "solo_confidence_threshold": 0.35,
        "prefer_engine_on_tie": "suryaocr"
    })

    combiner = OCRCombiner(
        engine_registry=engine_registry,
        engine_names=args_dict.get("ocr_engines", list(AVAILABLE_ENGINES.keys())),
        document_parser=doc_parser,
        document_path=document_path,
        lang_list=args_dict.get("lang", ["ar"]),
        engine_configs=engine_configs,
        merge_engine_outputs=merge_pairs if use_word_merging else [], # Pass pairs only if merging
        word_merger_config=merger_config if use_word_merging else {}
    )

    # This will hold the text data for each page, either List[List[str]] or List[str]
    page_level_ocr_content: Any = None

    if use_word_merging and merge_pairs:
        logger.info("Starting OCR processing with word merging...")
        merged_structured_output_per_page = combiner.run_ocr_and_merge() # Assumes this method exists and works
        
        temp_page_texts = []
        if merged_structured_output_per_page:
            for page_idx, page_data in enumerate(merged_structured_output_per_page):
                page_text = OCRCombiner.reassemble_text_from_structured(page_data)
                temp_page_texts.append(page_text)
                if args_dict.get("verbose", 0) >= 2:
                    logger.debug(f"--- Merged/Reassembled Page {page_idx+1} ---\n{page_text[:200]}...")
            page_level_ocr_content = temp_page_texts # List[str]
            logger.info("Word merging and text reassembly completed for all pages.")
        else:
            logger.warning("Word merging was enabled, but run_ocr_and_merge returned no data.")
            page_level_ocr_content = [] # Ensure it's a list
    else:
        logger.info("Starting OCR processing without word merging (raw engine outputs)...")
        # This returns List[List[str]] -> List of pages, each page is List of engine texts
        page_level_ocr_content = combiner.run_ocr_pipeline_parallel()
        logger.info("Raw OCR text processing completed for all pages.")

    # LLM Refinement Stage
    final_processed_content_list: List[str] = []

    if args_dict.get("use_llm", True):
        logger.info("LLM refinement is enabled.")
        if not args_dict.get("groq_api_key") and not os.environ.get("GROQ_API_KEY"):
            logger.warning("Groq API key not provided. LLM refinement will be skipped.")
            # If LLM is skipped, convert page_level_ocr_content to List[str] if it's List[List[str]]
            if page_level_ocr_content and isinstance(page_level_ocr_content[0], list): # Raw engine outputs
                final_processed_content_list = ["\n---\n".join(page_outputs) for page_outputs in page_level_ocr_content]
            else: # Already List[str] from merging or empty
                final_processed_content_list = page_level_ocr_content if page_level_ocr_content else []
        elif not page_level_ocr_content:
            logger.warning("No OCR content to refine with LLM.")
            final_processed_content_list = []
        else:
            try:
                groq_api_key = args_dict.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
                if not groq_api_key: # Should have been caught above, but double check
                     raise ValueError("Groq API Key missing for LLM processing")

                groq_client = GroqClient(model_name=args_dict.get("llm_model_name", "gemma2-9b-it"), api_token=groq_api_key)
                llm_processor = LLMProcessor(llm_client=groq_client)
                llm_model_kwargs = {"temperature": args_dict.get("llm_temp", 0.0)}
                lang_list_str_for_llm = ", ".join(args_dict.get("lang", ["ar"]))

                num_pages_to_refine = len(page_level_ocr_content)
                logger.info(f"Starting LLM refinement for {num_pages_to_refine} pages...")

                for i, ocr_input_for_page in enumerate(page_level_ocr_content):
                    page_num = i + 1
                    
                    # LLMProcessor.refine_text expects List[str] (multiple engine outputs for one page)
                    # If merging happened, ocr_input_for_page is already a single string for that page.
                    # If no merging, ocr_input_for_page is List[str] (raw outputs for that page).
                    llm_input_texts = [ocr_input_for_page] if isinstance(ocr_input_for_page, str) else ocr_input_for_page
                    
                    logger.info(f"Sending Page {page_num}/{num_pages_to_refine} to LLM ({args_dict.get('llm_model_name', 'gemma2-9b-it')}). Input text count: {len(llm_input_texts)}")

                    raw_llm_response = llm_processor.refine_text(
                        llm_input_texts,
                        lang_list_str=lang_list_str_for_llm,
                        context_keywords=args_dict.get("llm_context_keywords", ""),
                        model_kwargs=llm_model_kwargs
                    )
                    
                    if "[LLM_ERROR" not in raw_llm_response.upper():
                        # refined_page_text = llm_processor.parse_llm_output(raw_llm_response)
                        refined_page_text = raw_llm_response            # got rid of json formatting
                        if "[LLM_PARSE_ERROR" in refined_page_text.upper():
                            logger.error(f"LLM parsing failed for Page {page_num}. Using raw LLM response. Details: {refined_page_text}")
                            final_processed_content_list.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_PARSE_ERROR: {raw_llm_response}]")
                        else:
                            final_processed_content_list.append(refined_page_text)
                            logger.info(f"LLM refinement complete for Page {page_num}.")
                    else:
                        logger.error(f"LLM refinement API call failed for Page {page_num}. LLM response: {raw_llm_response}")
                        final_processed_content_list.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_API_ERROR: {raw_llm_response}]")
                logger.info("LLM refinement process completed.")
            except Exception as e:
                logger.error(f"Critical error during LLM processing: {e}. Output may be mixed or fallback.", exc_info=True)
                # Fallback if LLM fails catastrophically
                if page_level_ocr_content and isinstance(page_level_ocr_content[0], list):
                    final_processed_content_list = ["\n---\n".join(page_outputs) for page_outputs in page_level_ocr_content]
                else:
                    final_processed_content_list = page_level_ocr_content if page_level_ocr_content else []
    else: # LLM not used
        logger.info("LLM refinement is disabled.")
        if page_level_ocr_content and isinstance(page_level_ocr_content[0], list): # Raw engine outputs List[List[str]]
            # If no merging and no LLM, decide how to present raw outputs:
            # Option 1: Join them per page
            final_processed_content_list = []
            for i, page_outputs_list in enumerate(page_level_ocr_content):
                page_header = f"---- Page {i+1} (Raw Outputs) ----\n"
                page_content_parts = [page_header]
                for j, ocr_engine_out in enumerate(page_outputs_list):
                    engine_name = args_dict.get("ocr_engines", [])[j] if j < len(args_dict.get("ocr_engines", [])) else f"Engine {j+1}"
                    engine_header = f"Output from {engine_name}:\n"
                    page_content_parts.append(engine_header)
                    page_content_parts.append(ocr_engine_out + "\n")
                final_processed_content_list.append("".join(page_content_parts))

        elif page_level_ocr_content: # Already List[str] (e.g. from merging, but LLM disabled)
            final_processed_content_list = page_level_ocr_content
        else: # No content at all
            final_processed_content_list = []

        
    logger.info("Core OCR processing finished.")
    return final_processed_content_list


def main_cli():
    parser = argparse.ArgumentParser(description="EvenBetterOCR: Advanced OCR Processing Pipeline.")
    parser.add_argument("document_path", type=str, help="Path to the document (PDF or image file) to process.")
    parser.add_argument("--ocr_engines", nargs="+", type=str, default=list(AVAILABLE_ENGINES.keys()),
                        choices=list(AVAILABLE_ENGINES.keys()), help="List of OCR engines to use.")
    parser.add_argument("--lang", nargs="+", type=str, default=["ar"],
                        help="List of language codes for OCR (e.g., 'en' 'ar').")
    parser.add_argument("--engine_configs_json", type=str, default="{}",
                        help="JSON string with specific configurations for engines.")

    # New flag for word merging
    parser.add_argument("--use_word_merging", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable/Disable word-level merging of specified OCR engine outputs.")

    parser.add_argument("--use_llm", action=argparse.BooleanOptionalAction, default=True, help="Enable/Disable LLM refinement.")
    parser.add_argument("--llm_model_name", type=str, default="gemma2-9b-it", help="Groq LLM model name to use.")
    parser.add_argument("--groq_api_key", type=str, default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API key. Can also be set via GROQ_API_KEY environment variable.")
    parser.add_argument("--llm_context_keywords", type=str, default="", help="Optional context keywords for the LLM prompt.")
    parser.add_argument("--llm_temp", type=float, default=0.0, help="Temperature for LLM generation.")

    parser.add_argument("--display_bounding_boxes", type=str, choices=list(AVAILABLE_ENGINES.keys()) + [None], default=None, # Allow None
                        help="Display bounding boxes from a specific engine for the first page (requires GUI).")
    parser.add_argument("--display_annotated_output", type=str, choices=list(AVAILABLE_ENGINES.keys()) + [None], default=None, # Allow None
                        help="Display annotated output from a specific engine for the first page (requires GUI).")
    
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (-v for INFO, -vv for DEBUG).")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the final text output.")

    args = parser.parse_args()
    args_dict = vars(args) # Convert Namespace to dict for run_ocr_processing

    # GUI display logic (run this before core processing if needed, and if not in server context)
    # This part is problematic for server use and should ideally be separated or handled differently.
    # For now, it will be skipped if run_ocr_processing is called directly.
    if not os.environ.get("FLASK_RUNNING"): # Simple check if we are in CLI mode
        if args.display_bounding_boxes or args.display_annotated_output:
            display_engine_name = args.display_bounding_boxes or args.display_annotated_output
            if display_engine_name: # Check if a name is actually provided
                logger.info(f"Attempting to display output for engine: {display_engine_name}")
                try:
                    # This part needs access to engine_registry, doc_parser etc.
                    # which are inside run_ocr_processing. This suggests display
                    # might need to be a post-processing step or integrated differently.
                    # For CLI, it's fine if run_ocr_processing also handles this,
                    # but it would need the `Image` objects.
                    # For simplicity, this display logic might be better as a separate script/tool.
                    # Quick fix: Load doc_parser and engine_registry here for display if CLI
                    doc_parser_display = DocumentParser()
                    images = doc_parser_display.load_images_from_document(args.document_path)
                    if images:
                        engine_registry_display = EngineRegistry()
                        engine_cls_disp = AVAILABLE_ENGINES.get(display_engine_name)
                        if engine_cls_disp:
                            engine_registry_display.register_engine(display_engine_name, engine_cls_disp)
                            first_page_image = images[0]
                            engine_class_to_display = engine_registry_display.get_engine_class(display_engine_name)
                            
                            engine_configs_display_json = args_dict.get("engine_configs_json", "{}")
                            engine_configs_display = json.loads(engine_configs_display_json)
                            engine_config_display_specific = engine_configs_display.get(display_engine_name, {})

                            engine_to_display = engine_class_to_display(lang_list=args.lang, **engine_config_display_specific)

                            if args.display_bounding_boxes:
                                logger.info(f"Displaying bounding boxes for {display_engine_name} on the first page...")
                                engine_to_display.display_bounding_boxes(first_page_image.copy())
                            if args.display_annotated_output:
                                logger.info(f"Displaying annotated output for {display_engine_name} on the first page...")
                                engine_to_display.display_annotated_output(first_page_image.copy())
                        else:
                             logger.warning(f"Display engine '{display_engine_name}' not in AVAILABLE_ENGINES.")
                    else:
                        logger.warning("No images loaded, cannot perform display operation.")
                except Exception as e:
                    logger.error(f"Error during display operation for engine {display_engine_name}: {e}", exc_info=True)


    final_text_output = run_ocr_processing(args_dict)

    print("\n--- Final Processed Text (from main_cli) ---")
    print(final_text_output)
    print("--- End of Final Processed Text ---")

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