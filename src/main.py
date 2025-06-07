# betterOCR/main.py
import argparse
import json
import logging
import os
from typing import Any, List, Dict

from parsers.parser import DocumentParser
from combiner.combiner import OCRCombiner
from engines.EngineRegistry import EngineRegistry
from engines.concrete_implementations.easyOCR import EasyOCREngine
from engines.concrete_implementations.suryaOCR import SuryaOCREngine
from engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from llm.clients.groq_client import GroqClient
from llm.llm_processor import LLMProcessor
from PIL import Image # For display functions if called directly

# --- Logger Setup ---
# Basic configuration, can be moved to a logger_config.py for more complexity
logging.basicConfig(
    level=logging.INFO, # Default level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # Output to console
        # logging.FileHandler("betterocr.log") # Optionally log to a file
    ]
)
logger = logging.getLogger("BetterOCR_Main")
# --- ---

AVAILABLE_ENGINES = {
    # "easyocr": EasyOCREngine,
    "suryaocr": SuryaOCREngine,
    "tesseractocr": TesseractOCREngine
}

def setup_global_logging_level(verbose_level: int):
    if verbose_level == 1:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Logging level set to INFO.")
    elif verbose_level >= 2:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.info("Logging level set to DEBUG.")
        # Set specific loggers to DEBUG if needed, e.g., for noisy libraries
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO) # Example of quieting a noisy logger
    else: # Default or 0
        logging.getLogger().setLevel(logging.WARNING) # Less verbose default
        logger.warning("Logging level set to WARNING.")


def main():
    parser = argparse.ArgumentParser(description="EvenBetterOCR: Advanced OCR Processing Pipeline.")
    parser.add_argument("document_path", type=str, help="Path to the document (PDF or image file) to process.")
    parser.add_argument("--ocr_engines", nargs="+", type=str, default=list(AVAILABLE_ENGINES.keys()),
                        choices=list(AVAILABLE_ENGINES.keys()), help="List of OCR engines to use.")
    parser.add_argument("--lang", nargs="+", type=str, default=["ar"], 
                        help="List of language codes for OCR (e.g., 'en' 'ar'). For Tesseract, use 'eng+ara'.")
    parser.add_argument("--engine_configs_json", type=str, default="{}",
                        help="JSON string with specific configurations for engines, e.g., '{\"tesseractocr\": {\"tesseract_config\": \"--psm 6\"}}'")

    parser.add_argument("--use_llm", action=argparse.BooleanOptionalAction, default=True, help="Enable/Disable LLM refinement.")
    parser.add_argument("--llm_model_name", type=str, default="gemma2-9b-it", help="Groq LLM model name to use.") 
    parser.add_argument("--groq_api_key", type=str, default=os.environ.get("GROQ_API_KEY"), 
                        help="Groq API key. Can also be set via GROQ_API_KEY environment variable.")
    parser.add_argument("--llm_context_keywords", type=str, default="", help="Optional context keywords for the LLM prompt.")
    parser.add_argument("--llm_temp", type=float, default=0.0, help="Temperature for LLM generation (if supported by client).")


    parser.add_argument("--display_bounding_boxes", type=str, choices=list(AVAILABLE_ENGINES.keys()), default=None,
                        help="Display bounding boxes from a specific engine for the first page (requires GUI).")
    parser.add_argument("--display_annotated_output", type=str, choices=list(AVAILABLE_ENGINES.keys()), default=None,
                        help="Display annotated output (boxes and text) from a specific engine for the first page (requires GUI).")
    
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (-v for INFO, -vv for DEBUG).")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the final text output.")


    args = parser.parse_args()

    setup_global_logging_level(args.verbose)
    logger.info(f"Application started with arguments: {args}")

    engine_registry = EngineRegistry()
    for name, cls in AVAILABLE_ENGINES.items():
        if name in args.ocr_engines:
            engine_registry.register_engine(name, cls)

    doc_parser = DocumentParser()

    try:
        engine_configs = json.loads(args.engine_configs_json)
        if not isinstance(engine_configs, dict):
            raise ValueError("engine_configs_json must be a JSON object (dict).")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON string for --engine_configs_json: {e}. Using empty configs.")
        engine_configs = {}

    if args.display_bounding_boxes or args.display_annotated_output:
        display_engine_name = args.display_bounding_boxes or args.display_annotated_output
        logger.info(f"Attempting to display output for engine: {display_engine_name}")
        try:
            images = doc_parser.load_images_from_document(args.document_path)
            if images:
                first_page_image = images[0]
                engine_class_to_display = engine_registry.get_engine_class(display_engine_name)
                engine_config_display = engine_configs.get(display_engine_name, {})
                
                engine_to_display = engine_class_to_display(lang_list=args.lang, **engine_config_display)

                if args.display_bounding_boxes:
                    logger.info(f"Displaying bounding boxes for {display_engine_name} on the first page...")
                    engine_to_display.display_bounding_boxes(first_page_image.copy())
                if args.display_annotated_output:
                    logger.info(f"Displaying annotated output for {display_engine_name} on the first page...")
                    engine_to_display.display_annotated_output(first_page_image.copy())
            else:
                logger.warning("No images loaded, cannot perform display operation.")
        except Exception as e:
            logger.error(f"Error during display operation for engine {display_engine_name}: {e}", exc_info=True)

    merge_pairs = []
    if 'suryaocr' in args.ocr_engines and 'tesseractocr' in args.ocr_engines:
        merge_pairs.append(('suryaocr', 'tesseractocr')) # Define which pair to merge (Surya as A, Tesseract as B for WordMerger)

    merger_config = {"iou_threshold": 0.5, "solo_confidence_threshold": 0.35, "prefer_engine_on_tie": "suryaocr"}


    combiner = OCRCombiner(
        engine_registry=engine_registry,
        engine_names=args.ocr_engines,
        document_parser=doc_parser,
        document_path=args.document_path,
        lang_list=args.lang,
        engine_configs=engine_configs,
        merge_engine_outputs=merge_pairs, # Pass the pairs
        word_merger_config=merger_config   # Pass merger config
    )

    logger.info("Starting OCR processing and merging for all pages...")
    merged_structured_output_per_page = combiner.run_ocr_and_merge()
    
    final_texts_for_llm_or_output: List[str] = []
    if merged_structured_output_per_page:
        for page_idx, page_data in enumerate(merged_structured_output_per_page):
            # Reassemble text for LLM or final string output
            # You'll need a good line reconstruction logic here if needed
            # For now, just join texts.
            page_text = OCRCombiner.reassemble_text_from_structured(page_data)
            final_texts_for_llm_or_output.append(page_text)
            if args.verbose >=2:
                logger.debug(f"--- Merged/Processed Page {page_idx+1} (for LLM/Output) ---\n{page_text[:200]}...")

    combined_ocr_text_per_page = final_texts_for_llm_or_output
    final_processed_content: Any = combined_ocr_text_per_page 

    if args.use_llm:
        logger.info("LLM refinement is enabled.")
        if not args.groq_api_key:
            logger.warning("Groq API key not provided. LLM refinement will be skipped.")
        elif not combined_ocr_text_per_page or not combined_ocr_text_per_page[0] : # Check if there's content
            logger.warning("No OCR content to refine with LLM.")
        else:
            try:
                # Ensure GROQ_API_KEY is set if using GroqClient implicitly
                if not os.environ.get("GROQ_API_KEY") and not args.groq_api_key:
                    logger.error("GROQ_API_KEY environment variable must be set for GroqClient if --groq_api_key is not provided.")
                    raise ValueError("Groq API Key missing")
                
                # GroqClient will use args.groq_api_key if provided, else os.environ.get("GROQ_API_KEY")
                groq_client = GroqClient(model_name=args.llm_model_name, api_token=args.groq_api_key)
                llm_processor = LLMProcessor(llm_client=groq_client)
                llm_model_kwargs = {"temperature": args.llm_temp}
                lang_list_str_for_llm = ", ".join(args.lang)

                refined_texts_all_pages: List[str] = []
                num_pages_to_refine = len(combined_ocr_text_per_page)
                logger.info(f"Starting LLM refinement for {num_pages_to_refine} pages...")

                for i, page_engine_outputs in enumerate(combined_ocr_text_per_page):
                    page_num = i + 1
                    logger.info(f"Sending Page {page_num}/{num_pages_to_refine} to LLM ({args.llm_model_name}) for refinement...")
                    
                    raw_llm_response = llm_processor.refine_text(
                        page_engine_outputs, 
                        lang_list_str=lang_list_str_for_llm,
                        context_keywords=args.llm_context_keywords,
                        model_kwargs=llm_model_kwargs
                    )
                    
                    # Check for errors from refine_text or parsing errors
                    if "[LLM_ERROR" not in raw_llm_response.upper(): # General check for error markers
                        logger.info(f"Parsing LLM response for Page {page_num}...")
                        refined_page_text = llm_processor.parse_llm_output(raw_llm_response)
                        
                        if "[LLM_PARSE_ERROR" in refined_page_text.upper():
                             logger.error(f"LLM parsing failed for Page {page_num}. Using raw LLM response as fallback for this page. Details: {refined_page_text}")
                             refined_texts_all_pages.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_PARSE_ERROR: {raw_llm_response}]") # Or just refined_page_text which contains the error
                        else:
                            refined_texts_all_pages.append(refined_page_text)
                            logger.info(f"LLM refinement complete for Page {page_num}.")
                            if args.verbose >=2:
                                 logger.debug(f"--- LLM Refined Output (Page {page_num}) ---\n{refined_page_text}\n--- End ---")
                            else:
                                logger.info(f"LLM Refined Output (Page {page_num}, first 100 chars): {refined_page_text[:100]}...")
                    else:
                        logger.error(f"LLM refinement failed during API call for Page {page_num}. LLM response: {raw_llm_response}")
                        refined_texts_all_pages.append(f"[LLM_REFINE_FAILED_PAGE_{page_num}_API_ERROR: {raw_llm_response}]")
                
                final_processed_content = refined_texts_all_pages 
                logger.info("LLM refinement process completed for all applicable pages.")

            except Exception as e:
                logger.error(f"An critical error occurred during the LLM processing setup or loop: {e}. Output may be mixed or fallback to OCR only.", exc_info=True)
                # final_processed_content will retain combined_ocr_text_per_page or partially refined list
    else:
        logger.info("LLM refinement is disabled. Using combined OCR text per page.")

    print("\n--- Final Processed Text ---")
    output_string_for_file = ""

    if isinstance(final_processed_content, list) and final_processed_content:
        if isinstance(final_processed_content[0], str): # Case 1: List[str] (LLM refined texts per page)
            logger.info("Formatting final output from list of refined page texts.")
            page_texts_for_output = []
            for i, page_text_content in enumerate(final_processed_content):
                header = f"\n--- Page {i+1} ---\n"
                print(header.strip())
                print(page_text_content)
                page_texts_for_output.append(header + page_text_content)
            output_string_for_file = "\n".join(page_texts_for_output)

        elif isinstance(final_processed_content[0], list): # Case 2: List[List[str]] (raw from OCR combiner)
            logger.info("Formatting final output from list of pages with multiple engine texts.")
            page_blocks_for_output = []
            for i, page_outputs_list in enumerate(final_processed_content):
                page_header = f"---- Page {i+1} ----\n"
                current_page_block_parts = [page_header]
                print(page_header.strip())
                for j, ocr_engine_out in enumerate(page_outputs_list):
                    engine_name = args.ocr_engines[j] if j < len(args.ocr_engines) else f"Engine {j+1}"
                    engine_header = f"Output from {engine_name}:\n"
                    print(engine_header.strip())
                    print(ocr_engine_out)
                    current_page_block_parts.append(engine_header)
                    current_page_block_parts.append(ocr_engine_out + "\n")
                current_page_block_parts.append("-------------------------\n")
                page_blocks_for_output.append("".join(current_page_block_parts))
            output_string_for_file = "".join(page_blocks_for_output)
        else:
            logger.warning(f"Final processed content is a list, but its elements are of an unrecognized type: {type(final_processed_content[0])}. Attempting direct string conversion.")
            output_string_for_file = str(final_processed_content)
            print(output_string_for_file)
    elif final_processed_content: # Not a list but not None/empty
        logger.warning(f"Final processed content is not a list: {type(final_processed_content)}. Attempting direct string conversion.")
        output_string_for_file = str(final_processed_content)
        print(output_string_for_file)
    else: # Empty
        logger.warning("Final processed content is empty.")
        output_string_for_file = "[No content processed]"
        print(output_string_for_file)
        
    print("--- End of Final Processed Text ---")

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(output_string_for_file)
            logger.info(f"Final output saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write output to file {args.output_file}: {e}", exc_info=True)

    logger.info("BetterOCR processing finished.")

if __name__ == "__main__":
    main()