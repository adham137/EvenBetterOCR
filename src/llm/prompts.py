# --- Prompt Templates (Easily Editable) ---
TEXT_DETECTION_PROMPT_TEMPLATE = """Combine and correct OCR results {result_indexes_prompt}, using \\n for line breaks.
Language is in {lang_list_str}. Remove unintended noise. Refer to the [context] keywords.
Answer in plain string format, Do not add anything extra.
Here are the output of 2 different OCR engines delimited by their indicies: {result_prompt}
{optional_context_prompt}"""

BOX_DETECTION_PROMPT_TEMPLATE = """Combine and correct OCR data {result_indexes_prompt}.
Include many items as possible. Language is in {lang_list_str} (Avoid arbitrary translations). Remove unintended noise.{optional_context_prompt_text} Answer in the JSON format.
Ensure coordinates are integers (round based on confidence if necessary) and output in the same JSON format (indent=0): Array({{'box':[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],'text':str}}):
{result_prompt}
{optional_context_prompt_data}"""