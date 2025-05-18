from langchain_groq import ChatGroq
import os
class GroqLLM:
    def __init__(self, model_name="gemma2-9b-it", api_token=None):
        self.model_name = model_name
        self.api_token = api_token or os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(model_name=model_name, groq_api_key=self.api_token)
        
    def run(self, prompt, model_kwargs=None):
        ai_message = self.llm.invoke(prompt)
        return ai_message.content if hasattr(ai_message, "content") else str(ai_message)