import json
import os
from openai import OpenAI
from typing import Dict, Any

class SelfQueryParser:
    """
    Phase 7: Metadata Filtering.
    Uses Mistral 7B to extract a strict JSON filter dictionary from the user's natural language query.
    """
    def __init__(self, api_key: str = "lm-studio", base_url: str = "http://127.0.0.1:1234/v1", model: str = "mistral-7b-instruct-v0.1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Cache to save time on repeated parses
        self.cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'self_query_cache.json'))
        self.cache = self._load_cache()
        
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4)
            
    def parse_query(self, query: str) -> Dict[str, Any]:
        if query in self.cache:
            return self.cache[query]
            
        system_prompt = (
            "You are an expert Metadata extraction system. "
            "Your task is to analyze the user's query and extract a JSON object representing the data sources they want to search. "
            "The possible data sources are: slack, gmail, linear, google_drive, hubspot, fireflies, github, jira, confluence. "
            "If the user mentions 'slack', output {\"source\": \"slack\"}. "
            "If they mention 'drive', 'docs' or 'spreadsheet', output {\"source\": \"google_drive\"}. "
            "If they mention 'meeting', 'transcript' or 'call', output {\"source\": \"fireflies\"}. "
            "If they mention 'ticket', 'jira' or 'bug', output {\"source\": \"jira\"}. "
            "If they do not imply a specific source, output {\"source\": null}. "
            "Output ONLY valid JSON, nothing else. No markdown formatting."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting Mistral might sneak in
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            parsed = json.loads(content.strip())
            
            # Save to cache
            self.cache[query] = parsed
            self._save_cache()
            return parsed
            
        except Exception as e:
            print(f"Self-Query Parser Error: {e} - Raw Response: {content if 'content' in locals() else 'None'}")
            return {"source": None}
