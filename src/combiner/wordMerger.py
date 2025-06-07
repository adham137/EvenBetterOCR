class WordMerger:
    def __init__(self, iou_threshold=0.5, solo_confidence_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.solo_confidence_threshold = solo_confidence_threshold
        

    def calculate_iou(boxA, boxB):
        # box format: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    def _normalize_confidence(self, item, engine_name):
        # Example: Tesseract gives 0-100, Surya 0-1
        confidence = item['confidence']
        if engine_name == 'tesseractocr' and confidence > 1.0:
            return confidence / 100.0
        # add other engine normalizations if needed
        return confidence

    def merge_page_outputs(self, tesseract_items: list, surya_items: list) -> list:
        merged_output = []
        surya_matched_indices = set()

        for tess_item in tesseract_items:
            best_surya_match = None
            max_iou = 0.0
            best_surya_idx = -1

            tess_norm_conf = self._normalize_confidence(tess_item, 'tesseractocr')

            for i, surya_item in enumerate(surya_items):
                if i in surya_matched_indices:
                    continue
                
                iou = self._calculate_iou(tess_item['bbox'], surya_item['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_surya_match = surya_item
                    best_surya_idx = i
            
            if best_surya_match and max_iou >= self.iou_threshold:
                surya_matched_indices.add(best_surya_idx)
                surya_norm_conf = self._normalize_confidence(best_surya_match, 'suryaocr')

                if tess_norm_conf >= surya_norm_conf:
                    merged_output.append(tess_item)
                else:
                    merged_output.append(best_surya_match)
            elif tess_norm_conf >= self.solo_confidence_threshold:
                merged_output.append(tess_item)

        for i, surya_item in enumerate(surya_items):
            if i not in surya_matched_indices:
                surya_norm_conf = self._normalize_confidence(surya_item, 'suryaocr')
                if surya_norm_conf >= self.solo_confidence_threshold:
                    merged_output.append(surya_item)
        
        # TODO: Add post-processing for remaining overlaps in merged_output
        # merged_output = self._post_process_overlaps(merged_output)
        return merged_output

    # def _post_process_overlaps(self, items: list) -> list:
    #     # Logic to handle items that still overlap after initial merge
    #     # e.g., remove fully contained duplicates if text is same/substring
    #     # This can get complex
    #     return refined_items