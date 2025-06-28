

import json
import pprint
from typing import List
from PIL import Image
import easyocr
import numpy as np
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from src.engines.concrete_implementations.easyOCR import EasyOCREngine
from src.engines.concrete_implementations.tesseractOCR import TesseractOCREngine
from src.engines.concrete_implementations.suryaOCR import SuryaOCREngine
from src.parsers.parser import DocumentParser

from src.llm.clients.gemini_client import GeminiClient


# PDF_PATH = 'D:\\ASU\\sem 10\\GRAD PROJ\\Getting Data\\dataset\\pdfs\\2025_1738.pdf'
# parser = DocumentParser()
# images = parser.load_images_from_document(PDF_PATH)

# sOCR = SuryaOCREngine(['ar'])

# sOCR.display_layout_regions(images[1], save_path='C:\\Users\\Adham\\Downloads\\diagram images\\2_layout.png')
# sOCR.display_detected_text_lines(images[1], with_layout_filtering=False, save_path='C:\\Users\\Adham\\Downloads\\diagram images\\2_text.png')
# sOCR.display_detected_text_lines(images[1], with_layout_filtering=True, save_path='C:\\Users\\Adham\\Downloads\\diagram images\\2_final.png')

# tokens_A = {'tokens': ['مديريه', 'التضامن', 'الاجتماعي', 'بالقاهره'], 'token_infos': [{'text': 'مديريه', 'confidence': 0.9736999526168361}, {'text': 'التضامن', 'confidence': 0.9736999526168361}, {'text': 'الاجتماعي', 'confidence': 0.9736999526168361}, {'text': 'بالقاهره', 'confidence': 0.9736999526168361}], 'line_confidence': 0.9736999526168361, 'engine_name': 'suryaocr'}
# tokens_B = {'tokens': ['مديريه', 'التضامز', 'الاجتماعي', 'بالقاهره'], 'token_infos': [{'text': 'مديريه', 'confidence': 0.7475}, {'text': 'التضامز', 'confidence': 0.7475}, {'text': 'الاجتماعي', 'confidence': 0.7475}, {'text': 'بالقاهره', 'confidence': 0.7475}], 'line_confidence': 0.7475, 'engine_name': 'tesseractocr'}

# import edlib
# from typing import Dict, Tuple, Optional
# import difflib
# from itertools import zip_longest

# class LineMerger:
#     def __init__(self, dictionary=None, insertion_confidence_threshold=0.5, vocab_override_confidence=0.95):
#         # This is a mock-up of the class structure for the methods to run.
#         self.dictionary = set(dictionary) if dictionary else set()
#         self.dictionary_available = bool(self.dictionary)
#         self.insertion_confidence_threshold = insertion_confidence_threshold
#         self.vocab_override_confidence = vocab_override_confidence

#     def _is_word_in_dictionary(self, word: str) -> bool:
#         # A sample implementation of the dictionary check.
#         return word.lower() in self.dictionary

#     def _merge_aligned_tokens(self, token_A_info: Optional[Dict], token_B_info: Optional[Dict]) -> Tuple[Optional[str], float]:
#         word_A = token_A_info['text'] if token_A_info else None
#         conf_A = token_A_info['confidence'] if token_A_info else 0.0
#         word_B = token_B_info['text'] if token_B_info else None
#         conf_B = token_B_info['confidence'] if token_B_info else 0.0

#         if word_A is not None and word_B is not None:
#             if word_A == word_B: return word_A, max(conf_A, conf_B)
#             if self.dictionary_available:
#                 in_dict_A = self._is_word_in_dictionary(word_A)
#                 in_dict_B = self._is_word_in_dictionary(word_B)
#                 if in_dict_A and not in_dict_B: return word_A, max(conf_A, self.vocab_override_confidence)
#                 if not in_dict_A and in_dict_B: return word_B, max(conf_B, self.vocab_override_confidence)
#             return (word_A, conf_A) if conf_A >= conf_B else (word_B, conf_B)
#         elif word_A is not None:
#             return (word_A, conf_A) if conf_A >= self.insertion_confidence_threshold else (None, 0.0)
#         elif word_B is not None:
#             return (word_B, conf_B) if conf_B >= self.insertion_confidence_threshold else (None, 0.0)
#         return None, 0.0

#     def merge_single_line(self, line_A_processed: Dict, line_B_processed: Dict) -> Tuple[str, float]:
#         tokens_A = line_A_processed['tokens']
#         token_infos_A = line_A_processed['token_infos']
#         tokens_B = line_B_processed['tokens']
#         token_infos_B = line_B_processed['token_infos']

#         if not tokens_A and not tokens_B: return "", 0.0
#         if not tokens_A: return " ".join(tokens_B), line_B_processed['line_confidence']
#         if not tokens_B: return " ".join(tokens_A), line_A_processed['line_confidence']

#         merged_tokens_list = []
#         merged_token_confidences = []
        

#         s = difflib.SequenceMatcher(None, tokens_A, tokens_B, autojunk=False)
#         # get_opcodes format: (tag, i1, i2, j1, j2)
#         # tag is 'equal', 'replace', 'delete' (from A), 'insert' (into A from B)
#         for tag, i1, i2, j1, j2 in s.get_opcodes():
#             if tag == 'equal':
#                 for k in range(i2 - i1):
#                     token_A_info_for_op = token_infos_A[i1 + k]
#                     token_B_info_for_op = token_infos_B[j1 + k]
#                     merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, token_B_info_for_op)
#                     if merged_word is not None:
#                         merged_tokens_list.append(merged_word)
#                         merged_token_confidences.append(merged_conf)
#             elif tag == 'replace':
#                 len_A_segment = i2 - i1
#                 len_B_segment = j2 - j1
#                 min_len = min(len_A_segment, len_B_segment)
#                 for k in range(min_len):
#                     token_A_info_for_op = token_infos_A[i1 + k]
#                     token_B_info_for_op = token_infos_B[j1 + k]
#                     merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, token_B_info_for_op)
#                     if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)
                
#                 if len_A_segment > min_len: # Remainder in A are deletions from B's perspective
#                     for k_extra in range(min_len, len_A_segment):
#                         token_A_info_for_op = token_infos_A[i1 + k_extra]
#                         merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, None)
#                         if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)
#                 elif len_B_segment > min_len: # Remainder in B are insertions from A's perspective
#                     for k_extra in range(min_len, len_B_segment):
#                         token_B_info_for_op = token_infos_B[j1 + k_extra]
#                         merged_word, merged_conf = self._merge_aligned_tokens(None, token_B_info_for_op)
#                         if merged_word is not None: merged_tokens_list.append(merged_word); merged_token_confidences.append(merged_conf)

#             elif tag == 'delete': # Delete from A (A has words, B has a gap)
#                 for k in range(i2 - i1):
#                     token_A_info_for_op = token_infos_A[i1 + k]
#                     merged_word, merged_conf = self._merge_aligned_tokens(token_A_info_for_op, None)
#                     if merged_word is not None:
#                         merged_tokens_list.append(merged_word)
#                         merged_token_confidences.append(merged_conf)
#             elif tag == 'insert': # Insert from B (B has words, A has a gap)
#                 for k in range(j2 - j1):
#                     token_B_info_for_op = token_infos_B[j1 + k]
#                     merged_word, merged_conf = self._merge_aligned_tokens(None, token_B_info_for_op)
#                     if merged_word is not None:
#                         merged_tokens_list.append(merged_word)
#                         merged_token_confidences.append(merged_conf)
            
#             if not merged_tokens_list: # If difflib somehow results in nothing, fallback to higher conf line
#                 # logger.warning("difflib alignment also resulted in no merged tokens. Falling back to highest confidence line text.")
#                 if line_A_processed['line_confidence'] >= line_B_processed['line_confidence']:
#                     return " ".join(tokens_A), line_A_processed['line_confidence']
#                 else:
#                     return " ".join(tokens_B), line_B_processed['line_confidence']

#         final_text = " ".join(merged_tokens_list)
#         overall_line_confidence = sum(merged_token_confidences) / len(merged_token_confidences) if merged_token_confidences else 0.0
#         return final_text, overall_line_confidence
# lm = LineMerger()
# print(lm.merge_single_line(tokens_A, tokens_B))