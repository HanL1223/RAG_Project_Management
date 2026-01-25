"""
LangChain Document Loaders convert various data sources into Document objects.
Built-in loaders exist for PDFs, web pages, databases, etc.

For our Jira data, we create custom loaders that:
  1. Read our processed JSONL/CSV files
  2. Convert each record to a LangChain Document
  3. Apply our text building/cleaning utilities

LANGCHAIN PATTERN:
------------------
All loaders implement the BaseLoader interface with two methods:
  - load(): Returns all documents at once
  - lazy_load(): Generator that yields documents one at a time

lazy_load() is preferred for large datasets (memory efficient).

WHY CUSTOM LOADERS?
-------------------
- Our data has specific structure (issue_key, description, etc.)
- We apply domain-specific text processing
- We attach rich metadata for filtering and display
"""

import csv
import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from langchain_core.documents import Document

from src.core.models import JiraIssue
from src.core.text_utils import build_issue_text

logger = logging.getLogger(__name__)


class JiraIssueLoader(ABC):
    """
    Abstract base class for Jira issue loaders.
    
    The load() method is implemented in terms of lazy_load(),
    so subclasses only need to implement lazy_load().
    
    This follows the same pattern as LangChain's BaseLoader,
    """
    def lazy_load(self) ->Iterator[Document]:
            
            """
            Lazily load documents one at a time.
        
        This is a generator that yields Document objects.
        Memory efficient for large datasets.
        
        Yields:
            LangChain Document objects
            """
            raise NotImplementedError
    def load(self)->list[Document]:
          """
        Load all document into memory
        method  collects all documents from lazy_load().
        Use lazy_load() for large datasets to avoid memory issues.

        Returns:
            List of all Document objects

          """
          return list(self.lazy_load())
class JiraJsonlLoader(JiraIssueLoader):
    def __init__(self,
                   path: Path| str,
                   max_issues:Optional[int] = None,
                   include_comments:bool =True):
            self.path = path
            self.max_issues = max_issues
            self.include_comments = include_comments
            if not self.path.exists():
                  raise FileNotFoundError(f"Jsonl file ot found in {self.path}")
    def lazy_load(self) -> Iterator[Document]:
          issue_count = 0
          