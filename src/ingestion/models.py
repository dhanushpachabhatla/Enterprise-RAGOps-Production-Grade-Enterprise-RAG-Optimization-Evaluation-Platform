from pydantic import BaseModel, Field
from typing import Dict, Any

class Document(BaseModel):
    doc_id: str = Field(description="Unique document ID, typically starting with dsid_")
    source: str = Field(description="Source of the document, e.g., slack, gmail, etc.")
    content: str = Field(description="Raw text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata like filename, path, etc.")
