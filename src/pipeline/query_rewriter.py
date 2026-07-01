import time
import os
import json
from openai import OpenAI
from typing import Optional

class QueryRewriter:
    """
    Phase 6: Context Engineering.
    Connects to local LM Studio (Mistral 7B) to generate HyDE documents.
    Implements a JSON cache to avoid re-generating documents if the script is restarted.
    """
    def __init__(self, api_key: str = "lm-studio", base_url: str = "http://127.0.0.1:1234/v1", model: str = "mistral-7b-instruct-v0.1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Initialize Cache
        self.cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hyde_cache.json'))
        self.cache = self._load_cache()
        
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4)
        
    def generate_hyde_document(self, query: str) -> str:
        """
        Takes a raw, messy user query and prompts the LLM to generate a
        hypothetical document (HyDE) that answers it.
        """
        if query in self.cache:
            # Skip LM Studio entirely if we already generated it!
            return self.cache[query]
            
        system_prompt = (
            "You are an expert enterprise knowledge assistant. "
            "Your task is to write a hypothetical document snippet that perfectly answers the user's question. "
            "Write it exactly as it would appear in a formal company wiki, slack message, or Jira ticket. "
            "Do NOT include conversational filler like 'Here is the document:'. "
            "Write only the factual paragraph itself."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3, # Low temperature for more factual, document-like tone
                max_tokens=256   # Keep the generated chunk size similar to our semantic chunks
            )
            hyde_doc = response.choices[0].message.content.strip()
            
            # Save to cache immediately so we don't lose progress if it crashes!
            self.cache[query] = hyde_doc
            self._save_cache()
            
            return hyde_doc
            
        except Exception as e:
            print(f"LM Studio Error: {e}")
            # Fallback to the original query if LM Studio crashes
            return query
