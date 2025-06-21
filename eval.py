import glob
import json
from typing import List, Dict, Any
from PIL import Image
from jiwer import cer, compute_measures
import torch
import os
import numpy as np
from PIL import Image
import statistics

from src.main import AVAILABLE_ENGINES
from src.parsers.parser import DocumentParser

def calculate_cer(gt: str, pred: str) -> float:
    return cer(gt, pred)

def calculate_levenshtein(gt: str, pred: str) -> int:
    measures = compute_measures(gt, pred)
    return measures['substitutions'] + measures['insertions'] + measures['deletions']


def ocr_pdf_easyocr(images: List[Image.Image]):
    try:
        import easyocr
    except ImportError:
        print("EasyOCR library not found. Please install it: pip install easyocr")
        return None

    all_text = []
    try:
        reader = easyocr.Reader(['ar'], gpu=True)
        print("  EasyOCR Reader initialized.")

        for i, image in enumerate(images):

            img_np = np.array(image)
            results = reader.readtext(img_np, detail=0, paragraph= True)
            
            page_text = ""
            page_text = "\n".join(results)

            all_text.append(page_text)

        return "\n".join(all_text)

    except ImportError as ie:
        print(f"ImportError during EasyOCR processing: {ie}. Ensure all dependencies are installed.")
        return None
    except Exception as e:
        print(f"An error occurred during EasyOCR processing: {e}")
        return None

def ocr_pdf_tesseract(images:List[Image.Image]):

    try:
        import pytesseract
    except ImportError:
        print("pytesseract library not found. Please install it: pip install pytesseract")
        return None


    all_text = []

    try:
        
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='ara')
            all_text.append(text)

        
        return "\n".join(all_text)

    except ImportError as ie:
        print(f"ImportError during Tesseract processing: {ie}. Ensure all dependencies are installed.")
        return None
    except pytesseract.TesseractError as te: # Catch Tesseract specific errors
        print(f"TesseractError during processing: {te}")
        print("This might be due to missing language packs, incorrect Tesseract configuration, or a problematic image.")
        return None
    except Exception as e:
        print(f"An error occurred during Tesseract OCR processing: {e}")
        return None

def ocr_pdf_surya(images: List[Image.Image]):
    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        import gc

    except ImportError:
        print("Surya library not found. Please install it")
        return None

    all_page_texts = []
    try:
        pil_images = [img.convert("RGB") for img in images]

        detection_predictor = DetectionPredictor(dtype=torch.float32) 
        recognition_predictor = RecognitionPredictor(dtype=torch.float32)
        
        preds = recognition_predictor(pil_images, det_predictor=detection_predictor,detection_batch_size= 4, recognition_batch_size=10)

        for page_detection in preds:
            lines = page_detection.text_lines
            full_page_text = "\n".join([line.text for line in lines])
            all_page_texts.append(full_page_text)

        gc.collect()
        torch.cuda.empty_cache()

        return "\n".join(all_page_texts)

    except ImportError as ie:
        print(f"ImportError during Surya processing: {ie}. Ensure all dependencies are installed.")
        return None
    except Exception as e:
        print(f"An error occurred during Surya processing: {e}")
        return None

def ocr_pdf_evenBetterOCR(pdf_path):
    from src.main import run_ocr_processing

    args_dict = {}
    args_dict["document_path"] = pdf_path

    # Detector Engine
    args_dict["detector_engine"] = 'suryaocr'

    # Recognizer Engines (ocr_engines from form corresponds to recognizer_engine_names)
    ocr_engines_str = 'suryaocr'
    args_dict["ocr_engines"] = [eng.strip() for eng in ocr_engines_str.split(',') if eng.strip() and eng.strip() in AVAILABLE_ENGINES]



    # Languages
    lang_str = "ar"
    args_dict["lang"] = [l.strip() for l in lang_str.split(',') if l.strip()]

    args_dict["engine_configs_json"] = "{}"
    
    # Line Merging
    args_dict["use_line_merging"] = "true"
    args_dict["line_merger_config_json"] = "{\"min_wordfreq_for_dict_check\": 1e-8, \"insertion_confidence_threshold\": 0.5}"
    try:
        import json
        args_dict["line_merger_config"] = json.loads(args_dict["line_merger_config_json"])
    except json.JSONDecodeError as e:
        print(e)
        args_dict["line_merger_config"] = {}


    # LLM Settings
    args_dict["use_llm"] = True
    args_dict["llm_refinement_threshold"] = 0.95
    args_dict["llm_model_name"] = "gemma2-9b-it"
    args_dict["groq_api_key"] = os.environ.get("GROQ_API_KEY")
    args_dict["llm_context_keywords"] = ""
    args_dict["llm_temp"] = 0.0
    
    # Verbosity for server-side logs (might be set globally for the app)
    args_dict["verbose"] = 0 
    
    # Display options are N/A for server
    args_dict["display_bounding_boxes"] = None
    args_dict["display_annotated_output"] = None
    args_dict["display_layout_regions"] = False
    args_dict["display_detected_lines"] = False
    args_dict["output_file"] = None


    ocr_result_text = run_ocr_processing(args_dict)

    return "\n".join(ocr_result_text)
def evaluate_ocr_engines(pdf_dir: str,
                         gt_dir: str,
                         json_path: str = "ocr_results.json"):
    """
    Iterate over all PDFs in `pdf_dir`, match each to a ground-truth .txt in `gt_dir`
    (by filename without extension), run each OCR engine, compute per-file CER & LEV,
    and keep flushing results out to `json_path` after each PDF.  Finally computes
    per-engine mean/std and which engine has the lowest/highest std on each metric.
    """
    parser = DocumentParser()
    engines = {

        "surya":         lambda pdf, imgs: ocr_pdf_surya(imgs),
        "easyocr":       lambda pdf, imgs: ocr_pdf_easyocr(imgs),
        "tesseract":     lambda pdf, imgs: ocr_pdf_tesseract(imgs),
        "evenBetterOCR": lambda pdf, imgs: ocr_pdf_evenBetterOCR(pdf),
    }

    # will hold each file's per-engine metrics
    file_results = []

    # aggregated lists for final summary
    scores = { name: {"cer": [], "lev": []} for name in engines }

    def flush_to_json(include_summary: bool = False):
        """Write current file_results (and optional summary) to JSON file."""
        out = {"files": file_results}
        if include_summary:
            # compute summary only once here
            summary = {}
            for name, vals in scores.items():
                if not vals["cer"]:
                    continue
                summary[name] = {
                    "mean_cer": statistics.mean(vals["cer"]),
                    "std_cer":  statistics.pstdev(vals["cer"]),
                    "mean_lev": statistics.mean(vals["lev"]),
                    "std_lev":  statistics.pstdev(vals["lev"]),
                }
            # find lowest/highest std
            def extremum(key, fn):
                return fn(summary.items(), key=lambda kv: kv[1][key])[0]
            out.update({
                "per_engine":       summary,
                "lowest_std_cer":   extremum("std_cer", min),
                "highest_std_cer":  extremum("std_cer", max),
                "lowest_std_lev":   extremum("std_lev", min),
                "highest_std_lev":  extremum("std_lev", max),
            })
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(out, jf, indent=2, ensure_ascii=False)

    # process each PDF
    for pdf_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        gt_path = os.path.join(gt_dir, f"{base}_modified.txt")
        if not os.path.exists(gt_path):
            print(f"[WARN] No ground truth for {base}, skipping.")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()

        images = parser.load_images_from_document(pdf_path)

        # run each engine
        for name, fn in engines.items():
            try:
                pred = fn(pdf_path, images)
            except Exception as e:
                print(f"[ERROR] {name} failed on {base}: {e}")
                continue

            cer = calculate_cer(gt_text, pred)
            lev = calculate_levenshtein(gt_text, pred)
            print(f"({name}) on {base}: CER({cer:.4f})  LEV({lev})")

            # record per-file
            file_results.append({
                "file":      base,
                "engine":    name,
                "cer":       cer,
                "lev":       lev,
            })

            # record for summary
            scores[name]["cer"].append(cer)
            scores[name]["lev"].append(lev)

        # flush after each PDF
        flush_to_json(include_summary=False)

    # final flush including summary
    flush_to_json(include_summary=True)

    # also return the final summary for immediate use
    with open(json_path, "r", encoding="utf-8") as jf:
        return json.load(jf)
    

if __name__ == "__main__":
    PDF_PATH  = 'D:\\ASU\\sem 10\\GRAD PROJ\\Getting Data\\dataset\\pdfs'
    GT_PATH   = 'D:\\ASU\\sem 10\\GRAD PROJ\\Getting Data\\dataset\\gt_preprocessed_txt'
    JSON_PATH = 'data\\ocr_bench.json'
    stats = evaluate_ocr_engines(
       pdf_dir= PDF_PATH,
       gt_dir= GT_PATH,
       json_path= JSON_PATH
    )

    import pprint
    pprint.pprint(stats)
    
