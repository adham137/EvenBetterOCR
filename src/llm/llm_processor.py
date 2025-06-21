import logging
import json
import re
from typing import Dict, List, Any  
from .prompts import TEXT_DETECTION_PROMPT_TEMPLATE, BOX_DETECTION_PROMPT_TEMPLATE 

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, llm_client, prompt_key: str = "TEXT_DETECTION"):
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
            llm_response = self.llm_client.run(prompt)
            logger.info("Received response from LLM for one page.")
            logger.debug(f"LLM raw response for page (first 300 chars): {llm_response[:300]}...")
            return llm_response
        except Exception as e:
            logger.error(f"Error during LLM inference for page: {e}", exc_info=True)
            return f"[LLM_ERROR_PAGE_INFERENCE: {e}]" # Return error message as string

    def parse_llm_output(self, llm_raw_output: str) -> str:
        """
        Parses the LLM's JSON output to extract the 'data' field.
        Expected format: {"data": "<output_string>"}
        """
        logger.debug(f"Raw LLM output (first 200 chars): {llm_raw_output[:200]!r}")

        # 1) Try to pull out any JSON code fences first
        fenced = re.search(r"```json(.*?)```", llm_raw_output, re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else llm_raw_output

        # 2) Extract first balanced {...} block
        json_snip = LLMProcessor._extract_braced_block(candidate)
        if not json_snip:
            logger.warning("No JSON block found in LLM output.")
            return f"[LLM_PARSE_ERROR: No JSON block found. Raw: {llm_raw_output}]"

        # 3) Clean up common slip-ups
        json_str = LLMProcessor._clean_json_snippet(json_snip)
        logger.debug(f"Cleaned JSON candidate: {json_str!r}")

        # 4) Attempt parse
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Snippet was: {json_str!r}")
            return f"[LLM_PARSE_ERROR: JSONDecodeError - {e}. Snippet: {json_str}]"

        # 5) Extract and return
        val = payload.get("data")
        if isinstance(val, str):
            logger.info("Successfully parsed LLM data.")
            return val
        else:
            logger.warning(f"'data' missing or not a string in JSON: keys={list(payload)}")
            return f"[LLM_PARSE_ERROR: 'data' missing or not str in {json_str}]"
    
    @staticmethod
    def _extract_braced_block(text: str) -> str | None:
        """Return the first {...} block in `text` by counting braces, or None."""
        start = text.find('{')
        if start < 0:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None
    @staticmethod
    def _clean_json_snippet(js: str) -> str:
        # strip surrounding backticks or quotes
        
        js = js.strip("`\" \n")
        # remove trailing commas before a closing brace
        js = re.sub(r",\s*}", "}", js)
        return js
