import os
from typing import Iterator
from .models import Document

class DatasetReader:
    def __init__(self, base_dir: str):
        """
        Initialize the reader with the base directory of the dataset (e.g., all_documents/).
        """
        self.base_dir = base_dir

    def read_documents(self) -> Iterator[Document]:
        """
        Lazily traverse the dataset directory and yield Document objects.
        This prevents loading all documents into memory at once.
        """
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                
                # Extract source from the directory structure
                # We expect the structure to be base_dir/<source>/...
                rel_path = os.path.relpath(root, self.base_dir)
                # The first folder inside base_dir is the source category
                source = rel_path.split(os.sep)[0] if rel_path != "." else "unknown"

                # Extract doc_id from filename
                # Filename format: dsid_<uuid>__<descriptive-name>.txt
                if file.startswith("dsid_") and "__" in file:
                    doc_id = file.split("__")[0]
                else:
                    doc_id = file.replace(".txt", "")

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading file: {e}"

                metadata = {
                    "filename": file,
                    "filepath": os.path.relpath(file_path, self.base_dir)
                }

                yield Document(
                    doc_id=doc_id,
                    source=source,
                    content=content,
                    metadata=metadata
                )
