import json
from typing import List
from pydantic import BaseModel, Field

class BenchmarkQuestion(BaseModel):
    question: str
    gold_answer: str
    expected_doc_ids: List[str]
    answer_facts: List[str] = Field(default_factory=list)

def load_benchmark_dataset(jsonl_path: str) -> List[BenchmarkQuestion]:
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Handle expected_doc_ids which might be a string or list
            doc_ids = data.get("expected_doc_ids", [])
            if isinstance(doc_ids, str):
                doc_ids = [doc_ids]
                
            questions.append(BenchmarkQuestion(
                question=data["question"],
                gold_answer=data.get("gold_answer", ""),
                expected_doc_ids=doc_ids,
                answer_facts=data.get("answer_facts", [])
            ))
    return questions
