import logging
import json
import re
from typing import Dict, List, Any 
from .clients.groq_client import GroqClient 
from llm.prompts import TEXT_DETECTION_PROMPT_TEMPLATE, BOX_DETECTION_PROMPT_TEMPLATE 

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, llm_client: GroqClient, prompt_key: str = "TEXT_DETECTION"):
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

    def prepare_prompt_for_text_refinement(self, single_page_engine_outputs: List[str], lang_list_str: str, context_keywords: str = "") -> str:
        """
        Prepares the prompt for text detection and correction for a single page's outputs.
        """
        if not isinstance(single_page_engine_outputs, list):
            logger.error(f"Expected single_page_engine_outputs to be a list, got {type(single_page_engine_outputs)}")
            raise ValueError("single_page_engine_outputs must be a list of strings.")

        prompt_data = {
            "result_indexes_prompt": " ".join([f"[{i}]" for i in range(len(single_page_engine_outputs))]),
            "lang_list_str": lang_list_str,
            "result_prompt": "\n".join(f"[{i}] {t} [{i}]" for i, t in enumerate(single_page_engine_outputs)),
            "optional_context_prompt": f"[context]\n{context_keywords}\n[/context]" if context_keywords else ""
        }
        
        try:
            formatted_prompt = self.prompt_template.format(**prompt_data)
            logger.debug(f"LLM prompt prepared for single page (first 300 chars): {formatted_prompt[:300]}...")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Missing key in prompt template data for TEXT_DETECTION: {e}. Data provided: {prompt_data.keys()}")
            raise ValueError(f"Failed to format prompt due to missing key: {e}")

    def refine_text(self, single_page_engine_outputs: List[str], lang_list_str: str, context_keywords: str = "", model_kwargs: Dict = None) -> str:
        """
        Sends a single page's combined OCR output (from multiple engines) to the LLM for refinement.
        """
        if self.prompt_template != TEXT_DETECTION_PROMPT_TEMPLATE:
             logger.warning(f"Refining text with a prompt template ('{self.prompt_template[:30]}...') not designed for text detection. Results might be unexpected.")
        
        prompt = self.prepare_prompt_for_text_refinement(single_page_engine_outputs, lang_list_str, context_keywords)
        
        logger.info("Sending request to LLM for text refinement of one page...")
        try:
            llm_response = self.llm_client.run(prompt, model_kwargs=model_kwargs or {})
            logger.info("Received response from LLM for one page.")
            logger.debug(f"LLM raw response for page (first 300 chars): {llm_response[:300]}...")
            return llm_response
        except Exception as e:
            logger.error(f"Error during LLM inference for page: {e}", exc_info=True)
            return f"[LLM_ERROR_PAGE_INFERENCE: {e}]" # Return error message as string

    def parse_llm_output(self, llm_raw_output: str) -> str:
        """
        Parses the LLM's JSON output to extract the refined text.
        Expected format: {"data": "<output_string>"}
        """
        logger.debug(f"Attempting to parse LLM output (first 100 chars): {llm_raw_output[:100]}")
        try:
            
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else: # Fallback to finding first '{' and last '}'
                json_start = llm_raw_output.find('{')
                json_end = llm_raw_output.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    json_str = llm_raw_output[json_start:json_end]
                else:
                    logger.warning(f"Could not find valid JSON structure in LLM output: {llm_raw_output}")
                    return f"[LLM_PARSE_ERROR: No valid JSON found. Raw output: {llm_raw_output}]"

            logger.debug(f"Extracted JSON string for parsing: {json_str}")
            data = json.loads(json_str)
            if "data" in data and isinstance(data["data"], str):
                logger.info("LLM output parsed successfully.")
                return data["data"]
            else:
                logger.warning(f"LLM JSON output does not contain 'data' key with a string value. Found keys: {data.keys()}. JSON: {json_str}")
                return f"[LLM_PARSE_ERROR: 'data' field missing or not a string in {json_str}]"

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM output: {e}. Extracted JSON was: '{json_str if 'json_str' in locals() else 'Not Extracted'}' Raw output: {llm_raw_output}")
            return f"[LLM_PARSE_ERROR: JSONDecodeError - {e}. Raw output: {llm_raw_output}]"
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing LLM output: {e}", exc_info=True)
            return f"[LLM_PARSE_ERROR: Unexpected error - {e}. Raw output: {llm_raw_output}]"