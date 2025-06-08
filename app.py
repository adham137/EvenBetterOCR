import os
import json
import uuid
import tempfile
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from src.main import run_ocr_processing, AVAILABLE_ENGINES


app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'ocr_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Allowed extensions for PDF
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file part in the request"}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No PDF file selected"}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save it temporarily
        original_filename = secure_filename(file.filename)
        temp_filename = str(uuid.uuid4()) + "_" + original_filename
        temp_doc_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            file.save(temp_doc_path)
            app.logger.info(f"File saved temporarily to {temp_doc_path}")

            # --- Construct args_dict from request.form ---
            args_dict = {}
            args_dict["document_path"] = temp_doc_path

            # OCR Engines (comma-separated string to list)
            ocr_engines_str = request.form.get("ocr_engines", ",".join(AVAILABLE_ENGINES.keys()))
            args_dict["ocr_engines"] = [eng.strip() for eng in ocr_engines_str.split(',') if eng.strip()]
            
            # Languages (comma-separated string to list)
            lang_str = request.form.get("lang", "ar")
            args_dict["lang"] = [l.strip() for l in lang_str.split(',') if l.strip()]

            args_dict["engine_configs_json"] = request.form.get("engine_configs_json", "{}")
            
            # Boolean flags
            args_dict["use_word_merging"] = request.form.get("use_word_merging", "false").lower() == "true"
            args_dict["use_llm"] = request.form.get("use_llm", "true").lower() == "true"

            args_dict["llm_model_name"] = request.form.get("llm_model_name", "gemma2-9b-it")
            args_dict["groq_api_key"] = request.form.get("groq_api_key", os.environ.get("GROQ_API_KEY"))
            args_dict["llm_context_keywords"] = request.form.get("llm_context_keywords", "")
            
            try:
                args_dict["llm_temp"] = float(request.form.get("llm_temp", 0.0))
            except ValueError:
                return jsonify({"error": "Invalid value for llm_temp, must be a float."}), 400

            # Server-side logging verbosity (can be configured differently for server)
            args_dict["verbose"] = int(request.form.get("verbose", 0)) # Or set globally for server
            
            # Display options are ignored in server context
            args_dict["display_bounding_boxes"] = None
            args_dict["display_annotated_output"] = None
            args_dict["output_file"] = None # Output is via HTTP response

            app.logger.info(f"Processing OCR with args: {args_dict}")
            
            # Call the refactored processing function
            # Set an environment variable so main.py knows it's run by Flask (for display logic)
            os.environ["FLASK_RUNNING"] = "true"
            ocr_result_text = run_ocr_processing(args_dict)
            del os.environ["FLASK_RUNNING"]

            return jsonify({"status": "success", "ocr_output": ocr_result_text}), 200

        except Exception as e:
            app.logger.error(f"Error during OCR processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_doc_path):
                try:
                    os.remove(temp_doc_path)
                    app.logger.info(f"Temporary file {temp_doc_path} removed.")
                except Exception as e_clean:
                    app.logger.error(f"Error removing temporary file {temp_doc_path}: {e_clean}")
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Set up basic logging for Flask if not already configured by main.py's import
    if not app.debug: # Only if not in debug mode (which has its own logger)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.INFO)
    app.logger.info(f"Temporary upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5000)