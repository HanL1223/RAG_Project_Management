"""
WHAT ARE DOCUMENT LOADERS?
--------------------------
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
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from langchain_core.documents import Document

from src.core.models import JiraIssue
from src.core.text_utils import build_issue_text

logger = logging.getLogger(__name__)



#Base Loader Interface

class JiraIssueLoader(ABC):
    
    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError
    def load(self) ->list[Document]:
        return list(self.lazy_load())


class JiraJsonlLoader(JiraIssueLoader):
    def __init__ (
            self,
            path: Path | str,
            max_issues:Optional[int] = None,
            include_comments:bool = True,
    ):
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not fond at {self.path}")
    def lazy_load(self)-> Iterator[Document]:
        issue_count = 0
        with self.path.open('r',encoding = 'utf-8') as f:
            for line_num,line in enumerate(f,start=1):
                if self.max_issues and issue_count >= self.max_issues:
                    logger.info(f"Reach max issue limit {self.max_issues}")
                    break
                line = line.strip() 
                if not line:
                    continue
                try:
                    #Parse Json
                    data = json.loads(line)

                    #Crete domain model
                    issue = JiraIssue.from_dict(data)

                    #Skipping invalid issues (no issue key or summary)
                    if not issue.issue_key or not issue.summary:
                        logger.warning(f"Skipping line {line_num}: missing issue key or summary")
                        continue
                    
                    text = build_issue_text(
                        summary=issue.summary,
                        description=issue.description,
                        acceptance_criteria=issue.acceptance_criteria,
                        comments=issue.comments if self.include_comments else None,
                    )

                    doc = issue.to_langchain_document(text)
                    issue_count += 1
                    yield doc
                except JiraJsonlLoader as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}, {e}")
        logger.info(f"Loaded {issue_count} issue from {self.path}")

#CSV loaded


class JiraCsvLoader(JiraIssueLoader):
    def __init__(self,
                 path: Path | str,
                 max_issues: Optional[int] = None,
                 include_comments :bool = True
                 ):
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found at {self.path}")
    def _extract_comments(self,row:dict) -> list[str]:
        comments = []
        for key,value in row.items():
            if not key:
                continue
            if key.lower().startwith("comment") and value:
                value = str(value).strip()
                if value:
                    # Extract body from format like "date;author..."
                    parts = value.spliot(';',2)
                    if len(parts) == 3:
                        body = parts[2].strip()
                    else:
                        body = value
                    if body:
                        comments.append(body)
        return comments
    
    def lazy_load(self) -> Iterator[Document]:
        issue_count = 0

        with self.path.open('r',encoding = 'utf-8-sig',newline = "") as f:
            reader = csv.DictReader(f)

            for row_num,row in enumerate(reader,start=1):
                if self.max_issues and issue_count >= self.max_issues:
                    break
                try:
                    issue = JiraIssue.from_dict(row)
                    if not issue.issue_key or issue.summary:
                        logger.warnning(
                            f"Skipping row {row_num}: missing issue key or summary"
                        )
                        continue
                    comments = None
                    if self.include_comments:
                        comments = self._extract_comments(row)
                    text = build_issue_text(
                        summary=issue.summary,
                        description=issue.description,
                        acceptance_criteria=issue.acceptance_criteria,
                        comments=issue.comments if self.include_comments else None,
                    )
                    doc = issue.to_langchain_document(text)
                    issue_count += 1
                    yield doc
                except Exception as e:
                    logger.warning(f"Error processing row {row_num}: {e}")
            logger.info(f"Loaded {issue_count} issues from {self.path}")

#Factory function

def create_loader(
        path: Path | str,
        max_issues:Optional[int] = None,
        include_comments:bool = True,
) -> JiraIssueLoader:
    path = Path(path)
    extension = path.suffix.lower()
    if extension in {".jsonl", ".jsonlines"}:
        return JiraJsonlLoader(
            path = path,
            max_issues=max_issues,
            include_comments= include_comments
        )
    
    if extension == '.csv':
        return JiraCsvLoader(
          path = path,
          max_issues=max_issues,
          include_comments= include_comments 
        )
    #To add more loadding method as needed
    raise ValueError(
        f"Unsupported file extension: {extension}. "
        f"Supported: .jsonl, .jsonlines, .csv"
    )
                
    


        
        
    