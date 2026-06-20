import os
import pytest
from src.ingestion.parser import DatasetReader

def test_dataset_reader(tmp_path):
    # Create a mock directory structure
    slack_dir = tmp_path / "slack"
    slack_dir.mkdir()
    
    file1 = slack_dir / "dsid_12345__some_discussion.txt"
    file1.write_text("Hello slack!", encoding="utf-8")
    
    gmail_dir = tmp_path / "gmail"
    gmail_dir.mkdir()
    
    file2 = gmail_dir / "dsid_67890__important_email.txt"
    file2.write_text("Hello gmail!", encoding="utf-8")
    
    # Add a file in the root to test "unknown" source extraction behavior
    root_file = tmp_path / "dsid_99999__root_file.txt"
    root_file.write_text("Hello root!", encoding="utf-8")

    reader = DatasetReader(str(tmp_path))
    docs = list(reader.read_documents())
    
    assert len(docs) == 3
    
    # Check slack doc
    slack_doc = next(d for d in docs if d.source == "slack")
    assert slack_doc.doc_id == "dsid_12345"
    assert slack_doc.content == "Hello slack!"
    assert slack_doc.metadata["filename"] == "dsid_12345__some_discussion.txt"
    
    # Check gmail doc
    gmail_doc = next(d for d in docs if d.source == "gmail")
    assert gmail_doc.doc_id == "dsid_67890"
    assert gmail_doc.content == "Hello gmail!"
    assert gmail_doc.metadata["filename"] == "dsid_67890__important_email.txt"

    # Check root doc
    root_doc = next(d for d in docs if d.source == "unknown")
    assert root_doc.doc_id == "dsid_99999"
    assert root_doc.content == "Hello root!"
