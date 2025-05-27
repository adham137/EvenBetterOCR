import logging
import json
from typing import Dict, List
from .clients.groq_client import GroqClient
from llm.prompts import TEXT_DETECTION_PROMPT_TEMPLATE, BOX_DETECTION_PROMPT_TEMPLATE 

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, llm_client: GroqClient, prompt_key: str = "TEXT_DETECTION"):
        """
        Initializes the LLMProcessor.
        Args:
            llm_client: An instance of an LLM client (e.g., GroqClient).
            prompt_key: Key to select the prompt template from llm.prompts.
                        Defaults to "TEXT_DETECTION".
        """
        self.llm_client = llm_client
        self.prompt_templates = {
            "TEXT_DETECTION": TEXT_DETECTION_PROMPT_TEMPLATE, 
            "BOX_DETECTION": BOX_DETECTION_PROMPT_TEMPLATE 
        }
        if prompt_key not in self.prompt_templates:
            logger.error(f"Prompt key '{prompt_key}' not found. Available keys: {list(self.prompt_templates.keys())}")
            raise ValueError(f"Invalid prompt_key: {prompt_key}")
        self.prompt_template = self.prompt_templates[prompt_key]
        logger.info(f"LLMProcessor initialized with LLM client: {llm_client.__class__.__name__} and prompt key: {prompt_key}")

    def prepare_prompt_for_text_refinement(self, combined_ocr_output: List[str], lang_list_str: str, context_keywords: str = "") -> str:
        """
        Prepares the prompt for text detection and correction using the TEXT_DETECTION_PROMPT_TEMPLATE.
        Args:
            combined_ocr_output: The combined text output from the OCR engines.
            lang_list_str: String representing the list of languages (e.g., "Arabic, English").
            context_keywords: Optional context keywords to guide the LLM.
        Returns:
            The fully formatted prompt string.
        """
        
        prompt_data = {
            "result_indexes_prompt": " ".join([f"[{i}]" for i in range(len(combined_ocr_output[0]))]),
            "lang_list_str": lang_list_str,
            "result_prompt": "\n".join(f"[{i}] {t} [{i}]" for i, t in enumerate(combined_ocr_output[0])),
            "optional_context_prompt": f"[context]\n{context_keywords}\n[/context]" if context_keywords else ""
        }
        
        try:
            formatted_prompt = self.prompt_template.format(**prompt_data)
            logger.debug(f"LLM prompt prepared: {formatted_prompt[:300]}...")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Missing key in prompt template data for TEXT_DETECTION: {e}. Data provided: {prompt_data.keys()}")
            raise ValueError(f"Failed to format prompt due to missing key: {e}")


    def refine_text(self, combined_ocr_output: str, lang_list_str: str, context_keywords: str = "", model_kwargs: Dict = None) -> str:
        """
        Sends the combined OCR output to the LLM for refinement.
        Args:
            combined_ocr_output: The text to refine.
            lang_list_str: String representing the list of languages.
            context_keywords: Optional context.
            model_kwargs: Additional arguments for the LLM model.
        Returns:
            The raw string output from the LLM.
        """
        if self.prompt_template != TEXT_DETECTION_PROMPT_TEMPLATE: # [cite: 58, 59, 60]
             logger.warning(f"Refining text with a prompt template ('{self.prompt_template[:30]}...') not designed for text detection. Results might be unexpected.")
        
        prompt = self.prepare_prompt_for_text_refinement(combined_ocr_output, lang_list_str, context_keywords)
        
        logger.info("Sending request to LLM for text refinement...")
        try:
            llm_response = self.llm_client.run(prompt, model_kwargs=model_kwargs or {})
            logger.info("Received response from LLM.")
            logger.debug(f"LLM raw response: {llm_response[:300]}...")
            return llm_response
        except Exception as e:
            logger.error(f"Error during LLM inference: {e}")
            return f"[LLM_ERROR: {e}]" # Return error message as string

    def parse_llm_output(self, llm_raw_output: str) -> str:
        """
        Parses the LLM's JSON output to extract the refined text.
        Expected format: {"data": "<output_string>"} [cite: 60]
        Args:
            llm_raw_output: The raw string output from the LLM.
        Returns:
            The extracted refined text string.
        """
        logger.debug(f"Attempting to parse LLM output: {llm_raw_output[:100]}")
        try:
            json_start = llm_raw_output.find('{')
            json_end = llm_raw_output.rfind('}') + 1
            
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = llm_raw_output[json_start:json_end]
                logger.debug(f"Extracted JSON string for parsing: {json_str}")
                data = json.loads(json_str)
                if "data" in data and isinstance(data["data"], str):
                    logger.info("LLM output parsed successfully.")
                    return data["data"]
                else:
                    logger.warning(f"LLM JSON output does not contain 'data' key with a string value. Found keys: {data.keys()}")
                    return f"[LLM_PARSE_ERROR: 'data' field missing or not a string in {json_str}]"
            else:
                logger.warning(f"Could not find valid JSON structure in LLM output: {llm_raw_output}")
                # Fallback: return the raw output if it's not the expected JSON, but log it.
                return f"[LLM_PARSE_ERROR: No valid JSON found. Raw output: {llm_raw_output}]"

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM output: {e}. Output was: {llm_raw_output}")
            return f"[LLM_PARSE_ERROR: JSONDecodeError - {e}. Raw output: {llm_raw_output}]"
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing LLM output: {e}")
            return f"[LLM_PARSE_ERROR: Unexpected error - {e}. Raw output: {llm_raw_output}]"