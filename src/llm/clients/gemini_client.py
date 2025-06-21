import os
import requests

class GeminiClient:
    """
    A simple client for Google Gemini's Generative Language API.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash",
                 api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided; set GEMINI_API_KEY env var")
        self.endpoint = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model_name}:generateContent"
        )

    def run(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        resp = requests.post(
            self.endpoint,
            params={"key": self.api_key},
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()  # raise exception on HTTP error

        data = resp.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError("Unexpected response format") from e
