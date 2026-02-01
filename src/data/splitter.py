"""
LangChain Text Splitters for Jira Documents

WHY SPLIT TEXT?
---------------
Embedding models have token limits (typically 512-8192 tokens).
Long documents must be split into smaller chunks that:
  1. Fit within the embedding model's context window
  2. Are semantically coherent (don't split mid-sentence)
  3. Have some overlap for context continuity

LANGCHAIN TEXT SPLITTERS:
-------------------------
LangChain provides several splitters:
  - CharacterTextSplitter: Splits on character count
  - RecursiveCharacterTextSplitter: Tries multiple separators
  - TokenTextSplitter: Splits based on actual token count
  - MarkdownHeaderTextSplitter: Respects markdown structure

We use RecursiveCharacterTextSplitter because:
  - It tries to split on paragraphs first, then sentences, then words
  - This preserves semantic coherence
  - It's the recommended default in LangChain

JIRA-SPECIFIC CUSTOMIZATION:
----------------------------
Our Jira documents have structure (Summary, Description, etc.).
We customize the separators to respect this structure.

================================================================================
"""

import hashlib
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)