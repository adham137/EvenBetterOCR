# combiner/line_pair_merger.py
import logging
from typing import List, Dict, Any, Tuple, Optional
import edlib
import difflib # For fallback alignment

# camel-tools for Arabic processing
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_teh_marbuta_ar, normalize_alef_ar
from camel_tools.tokenizers.word import simple_word_tokenize

# wordfreq for dictionary check
from wordfreq import word_frequency

logger = logging.getLogger(__name__)

class LineROVERMerger:
    # __init__, _is_word_in_dictionary, _preprocess_line_data, _merge_aligned_tokens
    # remain the SAME as in the previous response (with wordfreq integration).
    # I'll re-include them here for completeness of the class.

    def __init__(self,
                 lang: str = "ar",
                 use_word_confidences_if_available: bool = True,
                 insertion_confidence_threshold: float = 0.4,
                 vocab_override_confidence: float = 0.97,
                 min_wordfreq_for_dict_check: float = 1e-7
                ):
        self.lang = lang
        self.use_word_confidences = use_word_confidences_if_available
        self.insertion_confidence_threshold = insertion_confidence_threshold
        self.vocab_override_confidence = vocab_override_confidence
        self.min_wordfreq_for_dict_check = min_wordfreq_for_dict_check
        self.dictionary_available = False

        if self.lang == "ar":
            self.normalize_fn = lambda text: normalize_alef_ar(
                                            normalize_alef_maksura_ar(
                                            normalize_teh_marbuta_ar(str(text or ''))))
            self.tokenize_fn = simple_word_tokenize
            try:
                word_frequency("كلمة", "ar"); self.dictionary_available = True
                logger.info("LinePairMerger: Arabic processing with camel-tools & wordfreq dictionary.")
            except Exception as e:
                logger.warning(f"LinePairMerger: wordfreq error for Arabic: {e}. Dictionary check disabled.")
        else:
            self.normalize_fn = lambda text: str(text or '')
            self.tokenize_fn = lambda text: str(text or '').split()
            try:
                word_frequency("the", self.lang); self.dictionary_available = True
                logger.info(f"LinePairMerger: lang '{self.lang}' with wordfreq dictionary.")
            except:
                logger.info(f"LinePairMerger: lang '{self.lang}' (basic tokenization, no wordfreq).")

    def _is_word_in_dictionary(self, word: str) -> bool:
        if not self.dictionary_available or not word: return False
        return word_frequency(word, self.lang) >= self.min_wordfreq_for_dict_check

    def _preprocess_line_data(self, line_data: Dict[str, Any], engine_name: str) -> Dict[str, Any]:
        raw_text = line_data.get('text', "")
        line_rec_confidence = float(line_data.get('text_confidence', 0.0))
        word_segments = line_data.get('words')
        normalized_text = self.normalize_fn(raw_text)
        tokens = self.tokenize_fn(normalized_text)
        token_infos = []
        if not tokens:
            return {'tokens': [], 'token_infos': [], 'line_confidence': line_rec_confidence, 'engine_name': engine_name}
        scaled_line_conf = line_rec_confidence * line_rec_confidence
        if self.use_word_confidences and word_segments and len(word_segments) == len(tokens):
            all_segments_have_conf = all('confidence' in ws and isinstance(ws['confidence'], (float,int)) for ws in word_segments)
            if all_segments_have_conf:
                for i, token_str in enumerate(tokens):
                    segment_conf = float(word_segments[i].get('confidence', line_rec_confidence))
                    combined_conf = line_rec_confidence * (segment_conf if segment_conf <=1 else segment_conf/100.0)
                    token_infos.append({'text': token_str, 'confidence': combined_conf})
            else:
                 for token_str in tokens: token_infos.append({'text': token_str, 'confidence': scaled_line_conf})
        else:
            for token_str in tokens: token_infos.append({'text': token_str, 'confidence': scaled_line_conf})
        return {'tokens': tokens, 'token_infos': token_infos, 'line_confidence': line_rec_confidence, 'engine_name': engine_name}

    def _merge_aligned_tokens(self, token_A_info: Optional[Dict], token_B_info: Optional[Dict]) -> Tuple[Optional[str], float]:
        word_A = token_A_info['text'] if token_A_info else None
        conf_A = token_A_info['confidence'] if token_A_info else 0.0
        word_B = token_B_info['text'] if token_B_info else None
        conf_B = token_B_info['confidence'] if token_B_info else 0.0

        if word_A is not None and word_B is not None:
            if word_A == word_B: return word_A, max(conf_A, conf_B)
            if self.dictionary_available:
                in_dict_A = self._is_word_in_dictionary(word_A)
                in_dict_B = self._is_word_in_dictionary(word_B)
                if in_dict_A and not in_dict_B: return word_A, max(conf_A, self.vocab_override_confidence)
                if not in_dict_A and in_dict_B: return word_B, max(conf_B, self.vocab_override_confidence)
            return (word_A, conf_A) if conf_A >= conf_B else (word_B, conf_B)
        elif word_A is not None:
            return (word_A, conf_A) if conf_A >= self.insertion_confidence_threshold else (None, 0.0)
        elif word_B is not None:
            return (word_B, conf_B) if conf_B >= self.insertion_confidence_threshold else (None, 0.0)
        return None, 0.0

    def merge_single_line(self, line_A_processed: Dict, line_B_processed: Dict) -> Tuple[str, float]:
        tokens_A = line_A_processed['tokens']
        token_infos_A = line_A_processed['token_infos']
        tokens_B = line_B_processed['tokens']
        token_infos_B = line_B_processed['token_infos']

        if not tokens_A and not tokens_B: return "", 0.0
        if not tokens_A: return " ".join(tokens_B), line_B_processed['line_confidence']
        if not tokens_B: return " ".join(tokens_A), line_A_processed['line_confidence']

        merged_tokens_list = []
        merged_token_confidences = []
        
        op_codes_edlib = None
        try:
            alignment_result_edlib = edlib.align(tokens_A, tokens_B, mode="NW", task="path", k=-1)
            op_codes_edlib = alignment_result_edlib.get("alignment")
        except Exception as e:
            logger.warning(f"edlib.align raised an exception: {e}. Will attempt fallback alignment.")
            op_codes_edlib = None # Ensure it's None to trigger fallback

        if op_codes_edlib:
            logger.debug("Using edlib alignment for line merge.")
            ptr_A, ptr_B = 0, 0
            for op in op_codes_edlib:
                token_A_info_for_op, token_B_info_for_op = None, None
                if op == 0: # edlib: Match/Mismatch - aligned
                    if ptr_A < len(token_infos_A): token_A_info_for_op = token_infos_A[ptr_A]
                    if ptr_B < len(token_infos_B): token_B_info_for_op = token_infos_B[ptr_B]
                    ptr_A += 1; ptr_B += 1
                elif op == 1: # edlib: Insertion in A (query) relative to B (target) => B has gap
                    if ptr_A < len(token_infos_A): token_A_info_for_op = token_infos_A[ptr_A]
                    ptr_A += 1
                elif op == 2: # edlib: Deletion in A (query) relative to B (target) => A has gap
                    if ptr_B < len(token_infos_B): token_B_info_for_op = token_infos_B[ptr_B]
                    ptr_B += 1
                
                merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, token_B_info_for_op)
                if merged_word is not None:
                    merged_tokens_list.append(merged_word)
                    merged_token_confidences.append(merged_conf)
        else: # Fallback to difflib.SequenceMatcher
            logger.warning("edlib alignment failed or not available. Using difflib.SequenceMatcher as fallback.")
            s = difflib.SequenceMatcher(None, tokens_A, tokens_B)
            # get_opcodes format: (tag, i1, i2, j1, j2)
            # tag is 'equal', 'replace', 'delete' (from A), 'insert' (into A from B)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'equal':
                    for k in range(i2 - i1):
                        token_A_info_for_op = token_infos_A[i1 + k] if (i1 + k) < len(token_infos_A) else None
                        token_B_info_for_op = token_infos_B[j1 + k] if (j1 + k) < len(token_infos_B) else None # Should be same as A
                        merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, token_B_info_for_op)
                        if merged_word is not None:
                            merged_tokens_list.append(merged_word)
                            merged_token_confidences.append(merged_conf)
                elif tag == 'replace':
                    # For each replaced item, effectively align A's token with B's token
                    len_A_segment = i2 - i1
                    len_B_segment = j2 - j1
                    # This requires aligning the sub-segments; for simplicity, process one by one if lengths match
                    # Or if lengths differ, it's a block of insertions/deletions within the replace
                    # SequenceMatcher 'replace' can be complex if lengths are different.
                    # We'll iterate through the minimum of these lengths, treating them as direct replaces,
                    # and the remainder as insertions/deletions.
                    min_len = min(len_A_segment, len_B_segment)
                    for k in range(min_len):
                        token_A_info_for_op = token_infos_A[i1+k] if (i1+k) < len(token_infos_A) else None
                        token_B_info_for_op = token_infos_B[j1+k] if (j1+k) < len(token_infos_B) else None
                        merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, token_B_info_for_op)
                        if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)
                    
                    if len_A_segment > min_len: # Deletions from B's perspective / Insertions from A
                        for k_extra in range(min_len, len_A_segment):
                            token_A_info_for_op = token_infos_A[i1+k_extra] if (i1+k_extra) < len(token_infos_A) else None
                            merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, None) # A vs gap
                            if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)
                    elif len_B_segment > min_len: # Insertions from B's perspective / Deletions from A
                         for k_extra in range(min_len, len_B_segment):
                            token_B_info_for_op = token_infos_B[j1+k_extra] if (j1+k_extra) < len(token_infos_B) else None
                            merged_word, merged_conf = self._merge_aligned_tokens(None, token_B_info_for_op) # gap vs B
                            if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)

                elif tag == 'delete': # Delete from A (A has words, B has a gap)
                    for k in range(i2 - i1):
                        token_A_info_for_op = token_infos_A[i1 + k] if (i1+k) < len(token_infos_A) else None
                        merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, None) # A vs gap
                        if merged_word is not None:
                            merged_tokens_list.append(merged_word)
                            merged_token_confidences.append(merged_conf)
                elif tag == 'insert': # Insert from B (B has words, A has a gap)
                    for k in range(j2 - j1):
                        token_B_info_for_op = token_infos_B[j1 + k] if (j1+k) < len(token_infos_B) else None
                        merged_word, merged_conf = self._merge_aligned_tokens(None, token_B_info_for_op) # gap vs B
                        if merged_word is not None:
                            merged_tokens_list.append(merged_word)
                            merged_token_confidences.append(merged_conf)
            
            if not merged_tokens_list: # If difflib somehow results in nothing, fallback to higher conf line
                logger.warning("difflib alignment also resulted in no merged tokens. Falling back to highest confidence line text.")
                if line_A_processed['line_confidence'] >= line_B_processed['line_confidence']:
                    return " ".join(tokens_A), line_A_processed['line_confidence']
                else:
                    return " ".join(tokens_B), line_B_processed['line_confidence']


        final_text = " ".join(merged_tokens_list)
        overall_line_confidence = sum(merged_token_confidences) / len(merged_token_confidences) if merged_token_confidences else 0.0
        return final_text, overall_line_confidence

    # merge_page_results, merge_document_results, calculate_average_document_confidence
    # remain THE SAME as in the previous response.
    def merge_page_results(
        self,
        page_results_engine_A: List[Dict[str, Any]],
        page_results_engine_B: List[Dict[str, Any]],
        engine_A_name: str = "engine_A",
        engine_B_name: str = "engine_B"
    ) -> List[Dict[str, Any]]:
        if len(page_results_engine_A) != len(page_results_engine_B):
            logger.warning(f"Line count mismatch for page between {engine_A_name} ({len(page_results_engine_A)}) "
                           f"and {engine_B_name} ({len(page_results_engine_B)}). Processing common lines by index.")
        
        num_common_lines = min(len(page_results_engine_A), len(page_results_engine_B))
        merged_page_output_list = []

        for i in range(num_common_lines):
            line_A_full_data = page_results_engine_A[i]
            line_B_full_data = page_results_engine_B[i]

            base_line_info = {
                'bbox': line_A_full_data.get('bbox'), 'label': line_A_full_data.get('label'),
                'confidence': line_A_full_data.get('confidence'),
                'text_line_confidence': line_A_full_data.get('text_line_confidence'),
                'position': line_A_full_data.get('position')
            }
            base_line_info = {k: v for k,v in base_line_info.items() if v is not None}

            line_A_processed = self._preprocess_line_data(line_A_full_data, engine_A_name)
            line_B_processed = self._preprocess_line_data(line_B_full_data, engine_B_name)

            merged_text, merged_text_conf = self.merge_single_line(line_A_processed, line_B_processed)

            final_line_item = {**base_line_info, 'text': merged_text, 'text_confidence': merged_text_conf}
            merged_page_output_list.append(final_line_item)
        
        def append_remaining(source_lines, start_idx, dest_list):
            for k_idx in range(start_idx, len(source_lines)): dest_list.append(source_lines[k_idx])

        if len(page_results_engine_A) > num_common_lines:
            append_remaining(page_results_engine_A, num_common_lines, merged_page_output_list)
        elif len(page_results_engine_B) > num_common_lines:
            append_remaining(page_results_engine_B, num_common_lines, merged_page_output_list)
            
        return merged_page_output_list

    def merge_document_results(
        self,
        doc_results_engine_A: List[List[Dict[str, Any]]], 
        doc_results_engine_B: List[List[Dict[str, Any]]], 
        engine_A_name: str = "engine_A",
        engine_B_name: str = "engine_B"
    ) -> List[List[Dict[str, Any]]]:
        if len(doc_results_engine_A) != len(doc_results_engine_B):
            logger.warning(f"Page count mismatch for document between {engine_A_name} ({len(doc_results_engine_A)}) "
                           f"and {engine_B_name} ({len(doc_results_engine_B)}). Processing common pages by index.")
        
        num_common_pages = min(len(doc_results_engine_A), len(doc_results_engine_B))
        merged_document_output = []

        for page_idx in range(num_common_pages):
            page_A_lines = doc_results_engine_A[page_idx]
            page_B_lines = doc_results_engine_B[page_idx]
            merged_page = self.merge_page_results(page_A_lines, page_B_lines, engine_A_name, engine_B_name)
            merged_document_output.append(merged_page)

        def append_remaining_pages(source_doc_pages, start_idx, dest_doc_list):
            for k_idx in range(start_idx, len(source_doc_pages)): dest_doc_list.append(source_doc_pages[k_idx])

        if len(doc_results_engine_A) > num_common_pages:
            append_remaining_pages(doc_results_engine_A, num_common_pages, merged_document_output)
        elif len(doc_results_engine_B) > num_common_pages:
            append_remaining_pages(doc_results_engine_B, num_common_pages, merged_document_output)
            
        return merged_document_output

    def calculate_average_document_confidence(self, merged_document_pages: List[List[Dict[str, Any]]]) -> float:
        if not merged_document_pages: return 0.0
        all_line_confidences = [
            line.get('text_confidence', 0.0) 
            for page_lines in merged_document_pages if page_lines 
            for line in page_lines if isinstance(line.get('text_confidence'), (float, int))
        ]
        return sum(all_line_confidences) / len(all_line_confidences) if all_line_confidences else 0.0