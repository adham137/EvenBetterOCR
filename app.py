import os
import json
import uuid
import tempfile # Not strictly needed if directly saving to UPLOAD_FOLDER
import logging
from flask import Flask, request, jsonify, send_from_directory # send_from_directory for potential future use
from werkzeug.utils import secure_filename

# Ensure src is in PYTHONPATH or use appropriate relative imports if app.py is outside src
# If app.py is at the same level as the 'src' directory:
from src.main import run_ocr_processing, AVAILABLE_ENGINES, DEFAULT_DETECTOR_ENGINE

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'ocr_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload size (adjust as needed)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'document_file' not in request.files: # Changed from 'pdf_file' for generality
        return jsonify({"error": "No 'document_file' part in the request"}), 400
    
    file = request.files['document_file']
    if file.filename == '':
        return jsonify({"error": "No document file selected"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Use a unique name for the saved file to avoid collisions
        unique_suffix = str(uuid.uuid4()).split('-')[0] # Short unique ID
        temp_filename = f"{unique_suffix}_{original_filename}"
        temp_doc_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            file.save(temp_doc_path)
            app.logger.info(f"File saved temporarily to {temp_doc_path}")

            # --- Construct args_dict from request.form for run_ocr_processing ---
            args_dict = {}
            args_dict["document_path"] = temp_doc_path

            # Detector Engine
            args_dict["detector_engine"] = request.form.get("detector_engine", DEFAULT_DETECTOR_ENGINE)
            if args_dict["detector_engine"] not in AVAILABLE_ENGINES:
                 app.logger.warning(f"Requested detector '{args_dict['detector_engine']}' not available, using default '{DEFAULT_DETECTOR_ENGINE}'.")
                 args_dict["detector_engine"] = DEFAULT_DETECTOR_ENGINE

            # Recognizer Engines (ocr_engines from form corresponds to recognizer_engine_names)
            default_recognizers = [name for name in AVAILABLE_ENGINES.keys()] # Default to all available
            ocr_engines_str = request.form.get("ocr_engines", ",".join(default_recognizers))
            args_dict["ocr_engines"] = [eng.strip() for eng in ocr_engines_str.split(',') if eng.strip() and eng.strip() in AVAILABLE_ENGINES]
            if not args_dict["ocr_engines"]: # If user provided empty or all invalid
                args_dict["ocr_engines"] = default_recognizers


            # Languages
            lang_str = request.form.get("lang", "ar")
            args_dict["lang"] = [l.strip() for l in lang_str.split(',') if l.strip()]

            args_dict["engine_configs_json"] = request.form.get("engine_configs_json", "{}")
            
            # Line Merging
            args_dict["use_line_merging"] = request.form.get("use_line_merging", "true").lower() == "true"
            args_dict["line_merger_config_json"] = request.form.get("line_merger_config_json", '{}') # Passed as JSON string
            # The main_cli parses this JSON string. run_ocr_processing expects the already parsed dict.
            # So, we parse it here for run_ocr_processing.
            try:
                args_dict["line_merger_config"] = json.loads(args_dict["line_merger_config_json"])
            except json.JSONDecodeError:
                app.logger.warning(f"Invalid JSON for line_merger_config_json. Using empty dict. Value: {args_dict['line_merger_config_json']}")
                args_dict["line_merger_config"] = {}


            # LLM Settings
            args_dict["use_llm"] = request.form.get("use_llm", "true").lower() == "true"
            args_dict["llm_refinement_threshold"] = float(request.form.get("llm_refinement_threshold", 0.80))
            args_dict["llm_model_name"] = request.form.get("llm_model_name", "gemma2-9b-it")
            args_dict["groq_api_key"] = request.form.get("groq_api_key", os.environ.get("GROQ_API_KEY"))
            args_dict["llm_context_keywords"] = request.form.get("llm_context_keywords", "")
            try:
                args_dict["llm_temp"] = float(request.form.get("llm_temp", 0.0))
            except ValueError:
                return jsonify({"error": "Invalid value for llm_temp, must be a float."}), 400

            # Verbosity for server-side logs (might be set globally for the app)
            args_dict["verbose"] = int(request.form.get("verbose", 0)) 
            
            # Display options are N/A for server
            args_dict["display_bounding_boxes"] = None
            args_dict["display_annotated_output"] = None
            args_dict["display_layout_regions"] = False
            args_dict["display_detected_lines"] = False
            args_dict["output_file"] = None # Output is via HTTP response

            app.logger.info(f"Processing OCR with args: {args_dict}")
            
            # Call the refactored processing function
            os.environ["FLASK_RUNNING"] = "true" # To suppress GUI displays in main.py's display logic
            ocr_result_text = run_ocr_processing(args_dict)
            del os.environ["FLASK_RUNNING"]

            # Return the processed text
            return jsonify({"status": "success", "processed_text": ocr_result_text}), 200

        except Exception as e:
            app.logger.error(f"Error during OCR processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Setup basic logging for Flask if not already configured
    if not app.debug and not app.logger.handlers: # Check if handlers are already added
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.INFO)
    
    app.logger.info(f"Flask app starting. Temporary upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5000) # Changed port to avoid conflict if main.py runs on 5000