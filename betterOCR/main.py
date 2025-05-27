# betterOCR/main.py
import argparse
import json
import logging
import os
from typing import List, Dict

from parsers.parser import DocumentParser
from engines.combiner import OCRCombiner
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
    "easyocr": EasyOCREngine,
    # "suryaocr": SuryaOCREngine,
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

    # 1. Initialize Engine Registry and Register Engines
    engine_registry = EngineRegistry()
    for name, cls in AVAILABLE_ENGINES.items():
        if name in args.ocr_engines: # Only register requested engines
            engine_registry.register_engine(name, cls)

    # 2. Initialize Document Parser
    doc_parser = DocumentParser()

    # Parse engine_configs_json
    try:
        engine_configs = json.loads(args.engine_configs_json)
        if not isinstance(engine_configs, dict):
            raise ValueError("engine_configs_json must be a JSON object (dict).")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON string for --engine_configs_json: {e}. Using empty configs.")
        engine_configs = {}


    # --- Optional: Display functions for a single engine on the first page ---
    if args.display_bounding_boxes or args.display_annotated_output:
        display_engine_name = args.display_bounding_boxes or args.display_annotated_output
        logger.info(f"Attempting to display output for engine: {display_engine_name}")
        try:
            images = doc_parser.load_images_from_document(args.document_path)
            if images:
                first_page_image = images[0]
                engine_class_to_display = engine_registry.get_engine_class(display_engine_name)
                engine_config_display = engine_configs.get(display_engine_name, {})
                
                # Tesseract lang needs to be 'eng+ara' not ['en','ar']
                display_lang_param = args.lang
                if display_engine_name == "tesseractocr":
                    display_lang_param = "+".join(args.lang)

                engine_to_display = engine_class_to_display(lang=display_lang_param, **engine_config_display)

                if args.display_bounding_boxes:
                    logger.info(f"Displaying bounding boxes for {display_engine_name} on the first page...")
                    engine_to_display.display_bounding_boxes(first_page_image.copy())
                if args.display_annotated_output:
                    logger.info(f"Displaying annotated output for {display_engine_name} on the first page...")
                    engine_to_display.display_annotated_output(first_page_image.copy())
            else:
                logger.warning("No images loaded, cannot perform display operation.")
        except Exception as e:
            logger.error(f"Error during display operation for engine {display_engine_name}: {e}")
        # After display, the user might want to exit or continue. For now, we continue.


    # 3. Initialize OCR Combiner
    combiner = OCRCombiner(
        engine_registry=engine_registry,
        engine_names=args.ocr_engines,
        document_parser=doc_parser,
        document_path=args.document_path,
        lang=args.lang, # Pass the list of languages directly
        engine_configs=engine_configs
    )

    # 4. Run OCR Pipeline
    logger.info("Starting OCR processing...")
    combined_ocr_text = combiner.run_ocr_pipeline_parallel()
    
    logger.info("Combined OCR text processing completed.")
    if args.verbose >=2 : # DEBUG
        logger.debug(f"--- Combined OCR Output ---\n{combined_ocr_text}\n--- End of Combined OCR Output ---")
    else:
        logger.info(f"Combined OCR Output (first 500 chars): {combined_ocr_text[:500]}...")


    final_text = combined_ocr_text

    # 5. LLM Refinement (if enabled)
    if args.use_llm:
        logger.info("LLM refinement is enabled.")
        if not args.groq_api_key:
            logger.warning("Groq API key not provided. LLM refinement will be skipped. "
                           "Set GROQ_API_KEY environment variable or use --groq_api_key argument.")
        else:
            try:
                groq_client = GroqClient(model_name=args.llm_model_name)
                llm_processor = LLMProcessor(llm_client=groq_client)
                
                # LLM model_kwargs (example, temperature)
                llm_model_kwargs = {"temperature": args.llm_temp}

                logger.info(f"Sending combined text to LLM ({args.llm_model_name}) for refinement...")
                lang_list_str_for_llm = ", ".join(args.lang) # e.g., "en, ar"
                
                raw_llm_response = llm_processor.refine_text(
                    combined_ocr_text, 
                    lang_list_str=lang_list_str_for_llm,
                    context_keywords=args.llm_context_keywords,
                    model_kwargs=llm_model_kwargs
                )
                print(raw_llm_response)
                exit
                if "[LLM_ERROR:" not in raw_llm_response:
                    logger.info("Parsing LLM response...")
                    final_text = llm_processor.parse_llm_output(raw_llm_response)
                    logger.info("LLM refinement complete.")
                    if args.verbose >=2: # DEBUG
                         logger.debug(f"--- LLM Refined Output ---\n{final_text}\n--- End of LLM Refined Output ---")
                    else:
                        logger.info(f"LLM Refined Output (first 500 chars): {final_text[:500]}...")
                else:
                    logger.error(f"LLM refinement failed. Using combined OCR text. LLM response: {raw_llm_response}")
                    final_text = combined_ocr_text # Fallback to combined text
            except Exception as e:
                logger.error(f"An error occurred during LLM processing: {e}. Using combined OCR text.")
                final_text = combined_ocr_text # Fallback
    else:
        logger.info("LLM refinement is disabled. Using combined OCR text.")

    # 6. Output Final Text
    print("\n--- Final Processed Text ---")
    print(final_text)
    print("--- End of Final Processed Text ---")

    if args.output_file:
        try:
            if not isinstance(final_text, str):
                new_final_text: List[str] = []
                for i, page in enumerate(final_text):
                    new_final_text.append(f"---- page {i} ----\n")
                    for j, ocr_out in enumerate(page):
                        new_final_text.append(f"OCR{j}: \n{ocr_out}\n")
                    new_final_text.append("-------------------------\n")
                final_text = "".join(new_final_text)
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info(f"Final output saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write output to file {args.output_file}: {e}")

    logger.info("BetterOCR processing finished.")

if __name__ == "__main__":
    # Ensure the current directory (betterOCR) is in PYTHONPATH if running scripts from outside
    # For example, if adham137-evenbetterocr/ is the root, and you run python betterOCR/main.py
    # import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()