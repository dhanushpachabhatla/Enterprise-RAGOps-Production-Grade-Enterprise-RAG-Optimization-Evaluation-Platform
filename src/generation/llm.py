from openai import OpenAI
from typing import List, Dict, Any

class BaselineLLM:
    def __init__(self, api_key: str = "lm-studio", base_url: str = "http://127.0.0.1:1234/v1", model: str = "mistral-7b-instruct-v0.1"):
        # We default to local LM Studio
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        # Build prompt
        context_text = "\n\n---\n\n".join([f"Source: {c['doc_id']}\n{c['text']}" for c in context_chunks])
        
        system_prompt = (
            "You are a helpful enterprise AI assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say 'I cannot find the answer in the documents.'"
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        return response.choices[0].message.content
