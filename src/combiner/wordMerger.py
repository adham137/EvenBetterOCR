import logging
# from scipy.optimize import linear_sum_assignment # For Hungarian algorithm (optional advanced)
import numpy as np # For Hungarian algorithm if used

logger = logging.getLogger(__name__)

class WordMerger:
    def __init__(self, iou_threshold=0.4, solo_confidence_threshold=0.3, 
                 confidence_epsilon=0.05, prefer_engine_on_tie=None): # e.g. 'suryaocr'
        self.iou_threshold = iou_threshold
        self.solo_confidence_threshold = solo_confidence_threshold
        self.confidence_epsilon = confidence_epsilon # For tie-breaking text if confidences are close
        self.prefer_engine_on_tie = prefer_engine_on_tie # If confidences are identical/very close, prefer this engine
        logger.info(f"WordMerger initialized with IoU threshold: {iou_threshold}, Solo conf: {solo_confidence_threshold}")

    def _calculate_iou(self, boxA, boxB):
        # Ensure boxes are valid lists/tuples of 4 numbers
        if not (isinstance(boxA, (list, tuple)) and len(boxA) == 4 and
                isinstance(boxB, (list, tuple)) and len(boxB) == 4):
            logger.warning(f"Invalid bounding box format for IoU calculation. BoxA: {boxA}, BoxB: {boxB}")
            return 0.0
        try:
            # box format: [x1, y1, x2, y2]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0: # Optimization
                return 0.0

            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            if boxAArea <= 0 or boxBArea <= 0: # Avoid division by zero for invalid boxes
                return 0.0

            union_area = float(boxAArea + boxBArea - interArea)
            if union_area == 0: # Should not happen if interArea > 0 and areas > 0
                return 0.0 
            
            iou = interArea / union_area
            return iou
        except Exception as e:
            logger.error(f"Error in IoU calculation for BoxA: {boxA}, BoxB: {boxB} - {e}")
            return 0.0


    def _normalize_confidence(self, item_confidence, engine_name):
        # Example: Tesseract gives 0-100, Surya 0-1
        # It's crucial that 'confidence' key exists and is a number.
        if not isinstance(item_confidence, (int, float)):
            logger.warning(f"Confidence for {engine_name} is not a number: {item_confidence}. Defaulting to 0.")
            return 0.0

        if engine_name == 'tesseractocr': # Tesseract often uses 0-100 for words
            if item_confidence > 1.0: # Check if it's already normalized by mistake elsewhere
                return item_confidence / 100.0
            elif item_confidence == -1: # Tesseract uses -1 for non-word segment confidences
                return 0.0 # Or handle as very low confidence
            return item_confidence # Assuming it might already be 0-1 from some tesseract wrappers
        elif engine_name == 'suryaocr': # Surya usually 0-1
            return item_confidence
        elif engine_name == 'easyocr': # EasyOCR usually 0-1
            return item_confidence
        # Add other engine normalizations if needed
        logger.warning(f"Unknown engine '{engine_name}' for confidence normalization. Returning as is.")
        return item_confidence

    def merge_page_outputs(self, items_A: list, engine_A_name: str, items_B: list, engine_B_name: str) -> list:
        """
        Merges structured outputs from two engines for a single page.
        """
        final_merged_items = []
        
        # Defensive copy if needed, or ensure inputs are not modified if they are used elsewhere.
        # For now, we assume items_A and items_B can be consumed.
        
        # Handle cases where one or both inputs are empty or not lists of dicts
        if not (isinstance(items_A, list) and all(isinstance(i, dict) for i in items_A)):
            logger.warning(f"Input items_A for {engine_A_name} is not a valid list of dicts. Content: {str(items_A)[:100]}")
            items_A = []
        if not (isinstance(items_B, list) and all(isinstance(i, dict) for i in items_B)):
            logger.warning(f"Input items_B for {engine_B_name} is not a valid list of dicts. Content: {str(items_B)[:100]}")
            items_B = []

        # Optimization: if one list is empty, just return the other (after confidence filtering)
        if not items_A:
            for item_b in items_B:
                norm_conf_b = self._normalize_confidence(item_b.get('confidence'), engine_B_name)
                if norm_conf_b >= self.solo_confidence_threshold:
                    final_merged_items.append({**item_b, 'original_engine': engine_B_name})
            return self._post_process_overlaps(final_merged_items) # Post-process even single-engine results
        if not items_B:
            for item_a in items_A:
                norm_conf_a = self._normalize_confidence(item_a.get('confidence'), engine_A_name)
                if norm_conf_a >= self.solo_confidence_threshold:
                    final_merged_items.append({**item_a, 'original_engine': engine_A_name})
            return self._post_process_overlaps(final_merged_items)


        # Using a cost matrix for Hungarian algorithm (Scipy) can be more robust for one-to-many.
        # For now, stick to greedy matching for simplicity, but acknowledge its limits.
        # Matched_indices for items_B to avoid reusing them.
        b_matched_indices = [False] * len(items_B)

        # Iterate through engine A's items and find best match in B
        for idx_a, item_a in enumerate(items_A):
            best_match_idx_b = -1
            max_iou = 0.0

            # Ensure item_a has necessary keys
            if not all(k in item_a for k in ['bbox', 'text', 'confidence']):
                logger.warning(f"Item from {engine_A_name} is malformed: {item_a}. Skipping.")
                continue
            
            norm_conf_a = self._normalize_confidence(item_a['confidence'], engine_A_name)

            for idx_b, item_b in enumerate(items_B):
                if b_matched_indices[idx_b]:
                    continue # Already matched this item from B

                if not all(k in item_b for k in ['bbox', 'text', 'confidence']):
                    logger.warning(f"Item from {engine_B_name} is malformed: {item_b}. Skipping match attempt.")
                    continue

                iou = self._calculate_iou(item_a['bbox'], item_b['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_match_idx_b = idx_b
            
            chosen_item = None
            if best_match_idx_b != -1 and max_iou >= self.iou_threshold:
                b_matched_indices[best_match_idx_b] = True
                item_b_matched = items_B[best_match_idx_b]
                norm_conf_b = self._normalize_confidence(item_b_matched['confidence'], engine_B_name)

                # Decision logic for matched pair
                if abs(norm_conf_a - norm_conf_b) < self.confidence_epsilon:
                    # Confidences are very close - tie-breaking
                    # 1. Prefer specified engine if any
                    if self.prefer_engine_on_tie == engine_A_name:
                        chosen_item = {**item_a, 'original_engine': engine_A_name}
                    elif self.prefer_engine_on_tie == engine_B_name:
                        chosen_item = {**item_b_matched, 'original_engine': engine_B_name}
                    # 2. (Future) Consider text length/similarity or other heuristics
                    # For now, if no preference, default to A or the one with slightly higher raw confidence
                    elif norm_conf_a >= norm_conf_b: # Default to A on true tie
                        chosen_item = {**item_a, 'original_engine': engine_A_name}
                    else:
                        chosen_item = {**item_b_matched, 'original_engine': engine_B_name}
                elif norm_conf_a > norm_conf_b:
                    chosen_item = {**item_a, 'original_engine': engine_A_name}
                else:
                    chosen_item = {**item_b_matched, 'original_engine': engine_B_name}
                
                # Handling for "One-to-Many / Many-to-One Matches" (Partial)
                # If chosen_item's text is much shorter than the other item's text,
                # and the other item had decent confidence, this might be a split.
                # This is complex. For now, the above logic is a simple word-for-word choice.

            elif norm_conf_a >= self.solo_confidence_threshold: # Item A has no good match
                chosen_item = {**item_a, 'original_engine': engine_A_name}
            
            if chosen_item:
                final_merged_items.append(chosen_item)

        # Add remaining unmatched items from B if their confidence is good
        for idx_b, item_b in enumerate(items_B):
            if not b_matched_indices[idx_b]:
                if not all(k in item_b for k in ['bbox', 'text', 'confidence']):
                    logger.warning(f"Unmatched item from {engine_B_name} is malformed: {item_b}. Skipping.")
                    continue

                norm_conf_b = self._normalize_confidence(item_b.get('confidence'), engine_B_name)
                if norm_conf_b >= self.solo_confidence_threshold:
                    final_merged_items.append({**item_b, 'original_engine': engine_B_name})
        
        return self._post_process_overlaps(final_merged_items)

    def _post_process_overlaps(self, items: list) -> list:
        """
        Handles remaining overlaps in the merged list.
        - Removes fully contained duplicates if text is similar.
        - Sorts by reading order.
        """
        if not items:
            return []

        # Sort by y1 then x1 for reading order (simplistic, assumes no rotated text)
        # Bbox presence and format checked in _calculate_iou, but double check here.
        items.sort(key=lambda item: (item['bbox'][1] if isinstance(item.get('bbox'), list) and len(item['bbox'])==4 else float('inf'), 
                                     item['bbox'][0] if isinstance(item.get('bbox'), list) and len(item['bbox'])==4 else float('inf')))

        # Advanced: Remove highly overlapping items if they are essentially duplicates
        # This requires careful consideration of what constitutes a "better" duplicate.
        # For now, this step is a placeholder for more advanced logic.
        # A simple check: if two items have IoU > 0.9 and similar text, keep higher confidence.
        
        # Example of basic duplicate removal (very simplistic)
        # A more robust approach would build a graph of overlapping components.
        refined_items = []
        is_removed = [False] * len(items)
        for i in range(len(items)):
            if is_removed[i]:
                continue
            item_i = items[i]
            for j in range(i + 1, len(items)):
                if is_removed[j]:
                    continue
                item_j = items[j]
                
                iou = self._calculate_iou(item_i['bbox'], item_j['bbox'])
                if iou > 0.85: # High overlap, likely duplicate or one contains other
                    # This is where sophisticated logic for one-to-many merges could go.
                    # E.g., if item_i's text is "HelloWorld" and item_j's is "Hello"
                    # and item_j is mostly contained in item_i.
                    
                    # Simple rule: if IoU is high, prefer higher confidence.
                    # This could incorrectly remove a part of a split word.
                    conf_i = self._normalize_confidence(item_i.get('confidence'), item_i.get('original_engine', ''))
                    conf_j = self._normalize_confidence(item_j.get('confidence'), item_j.get('original_engine', ''))

                    # If text is identical, keep one with higher confidence
                    if item_i.get('text') == item_j.get('text'):
                        if conf_i >= conf_j:
                            is_removed[j] = True
                        else:
                            is_removed[i] = True
                            break # item_i is removed, move to next i
                    # Add more rules here: e.g. text similarity, containment checks...
            
            if not is_removed[i]:
                 refined_items.append(item_i)
        
        return refined_items