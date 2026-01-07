"""Enhanced metadata extraction module for RAG pipeline.

This module extracts rich metadata from documents to improve retrieval quality:
- Structure metadata (sections, headings, document type)
- Semantic metadata (topics, keywords, named entities)
- Technical metadata (code blocks, tables, equations)
- Quality signals (word count, reading level)

Usage:
    from utils.metadata_extractor import DocumentMetadataExtractor

    extractor = DocumentMetadataExtractor()
    metadata = extractor.extract_all_metadata(text, doc_format="pdf")

    # Or use convenience function for build_nodes() integration
    from utils.metadata_extractor import enhance_node_metadata
    enhanced = enhance_node_metadata(text, base_metadata)

Environment Variables:
    EXTRACT_ENHANCED_METADATA: Enable enhanced metadata extraction (0 or 1)
    EXTRACT_TOPICS: Enable topic extraction with TF-IDF (0 or 1)
    EXTRACT_ENTITIES: Enable named entity recognition (0 or 1)
    EXTRACT_CODE_BLOCKS: Detect code blocks in documents (0 or 1)
    EXTRACT_TABLES: Detect table structures (0 or 1)
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import Counter
from dataclasses import dataclass, field

# Optional NLP dependencies (graceful fallback if not available)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.debug("NLTK not available, using regex-based extraction")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.debug("scikit-learn not available, topic extraction disabled")

log = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Complete metadata extracted from a document or chunk.

    Attributes:
        structure: Structural information (sections, headings, doc type)
        semantic: Semantic information (topics, keywords, entities)
        technical: Technical content (code, tables, equations)
        quality: Quality signals (word count, reading level)
    """
    structure: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    technical: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for TextNode metadata."""
        flat = {}

        # Flatten with prefixes to avoid key collisions
        for k, v in self.structure.items():
            flat[f"struct_{k}"] = v
        for k, v in self.semantic.items():
            flat[f"sem_{k}"] = v
        for k, v in self.technical.items():
            flat[f"tech_{k}"] = v
        for k, v in self.quality.items():
            flat[f"qual_{k}"] = v

        return flat


class DocumentMetadataExtractor:
    """Extract rich metadata from documents to improve RAG retrieval.

    This class provides multiple extraction methods that can be enabled/disabled
    via environment variables. It uses lightweight NLP techniques (regex, NLTK, TF-IDF)
    to avoid heavy dependencies.

    Examples:
        >>> extractor = DocumentMetadataExtractor()
        >>> text = "# Introduction\\n\\nThis is a tutorial about Python..."
        >>> metadata = extractor.extract_all_metadata(text, doc_format="md")
        >>> print(metadata.structure["doc_type"])
        'tutorial'
    """

    # Common technical terms for entity extraction
    TECH_ENTITIES = {
        "python", "java", "javascript", "typescript", "rust", "go", "c++", "c#",
        "react", "vue", "angular", "django", "flask", "fastapi", "pytorch",
        "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
        "kubernetes", "docker", "aws", "gcp", "azure", "heroku",
        "postgresql", "mongodb", "redis", "mysql", "sqlite", "elasticsearch",
        "git", "github", "gitlab", "bitbucket", "jenkins", "circleci",
        "linux", "ubuntu", "debian", "centos", "macos", "windows"
    }

    # Document type indicators
    DOC_TYPE_PATTERNS = {
        "research_paper": [r"\babstract\b", r"\bconclusion\b", r"\breferences\b", r"\bcitation\b", r"\bfigure \d+\b"],
        "manual": [r"\binstallation\b", r"\bconfiguration\b", r"\btroubleshooting\b", r"\bsetup\b", r"\bappendix\b"],
        "tutorial": [r"\bstep \d+\b", r"\bexample\b", r"\btutorial\b", r"\bwalkthrough\b", r"\bguide\b"],
        "api_doc": [r"\bendpoint\b", r"\bapi\b", r"\bparameter\b", r"\brequest\b", r"\bresponse\b", r"\bstatus code\b"],
        "code": [r"def\s+\w+\(", r"class\s+\w+", r"function\s+\w+\(", r"import\s+\w+", r"package\s+\w+"],
        "blog_post": [r"\bposted\b", r"\bauthor\b", r"\bcomments\b", r"\bshare\b", r"\btags?\b"],
    }

    # Code block patterns
    CODE_PATTERNS = [
        r"```[\s\S]+?```",  # Markdown fenced code blocks
        r"~~~[\s\S]+?~~~",  # Alternative markdown code blocks
        r"(?:^|\n)(?:    |\t).+(?:\n(?:    |\t).+)*",  # Indented code blocks
        r"<code>[\s\S]+?</code>",  # HTML code tags
        r"<pre>[\s\S]+?</pre>",  # HTML pre tags
    ]

    # Table patterns
    TABLE_PATTERNS = [
        r"\|.+\|(?:\n\|[-:| ]+\|)+(?:\n\|.+\|)*",  # Markdown tables
        r"<table[\s\S]+?</table>",  # HTML tables
        r"(?:\n|^)(?:\S+\s+){2,}\S+(?:\n(?:\S+\s+){2,}\S+)*",  # Space-separated columns
    ]

    # Math equation patterns
    EQUATION_PATTERNS = [
        r"\$\$[\s\S]+?\$\$",  # Display math (LaTeX)
        r"\$[^\$\n]+?\$",  # Inline math (LaTeX)
        r"\\\[[\s\S]+?\\\]",  # Display math (alternative)
        r"\\\([\s\S]+?\\\)",  # Inline math (alternative)
    ]

    def __init__(self):
        """Initialize the metadata extractor.

        Downloads required NLTK data if not available (punkt, stopwords).
        """
        self.enabled = os.getenv("EXTRACT_ENHANCED_METADATA", "1") == "1"
        self.extract_topics = os.getenv("EXTRACT_TOPICS", "1") == "1"
        self.extract_entities = os.getenv("EXTRACT_ENTITIES", "1") == "1"
        self.extract_code = os.getenv("EXTRACT_CODE_BLOCKS", "1") == "1"
        self.extract_tables = os.getenv("EXTRACT_TABLES", "1") == "1"

        # Initialize NLTK data (download if needed)
        if NLTK_AVAILABLE and self.enabled:
            self._ensure_nltk_data()

        log.debug(f"DocumentMetadataExtractor initialized (enabled={self.enabled})")

    def _ensure_nltk_data(self):
        """Download required NLTK data if not available."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            log.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            log.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)

    def extract_all_metadata(
        self,
        text: str,
        doc_format: str = "txt",
        chunk_position: Optional[Tuple[int, int]] = None,
        section_title: Optional[str] = None,
    ) -> DocumentMetadata:
        """Extract all metadata from text.

        Args:
            text: Text content to analyze
            doc_format: Document format (pdf, html, md, py, etc.)
            chunk_position: Optional (current_chunk, total_chunks) tuple
            section_title: Optional section title if known

        Returns:
            DocumentMetadata object with all extracted metadata
        """
        if not self.enabled:
            return DocumentMetadata()

        metadata = DocumentMetadata()

        # Extract all metadata types
        metadata.structure = self.extract_structure_metadata(
            text, doc_format, chunk_position, section_title
        )
        metadata.semantic = self.extract_semantic_metadata(text)
        metadata.technical = self.extract_technical_metadata(text, doc_format)
        metadata.quality = self.extract_quality_signals(text)

        return metadata

    def extract_structure_metadata(
        self,
        text: str,
        doc_format: str = "txt",
        chunk_position: Optional[Tuple[int, int]] = None,
        section_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structural metadata from text.

        Detects:
        - Section titles (H1, H2, H3 headings)
        - Document type classification
        - Chunk position within document
        - Hierarchical structure

        Args:
            text: Text content to analyze
            doc_format: Document format (pdf, html, md, py, etc.)
            chunk_position: Optional (current_chunk, total_chunks) tuple
            section_title: Optional section title if known

        Returns:
            Dictionary with structure metadata
        """
        structure = {
            "format": doc_format,
            "has_headings": False,
            "heading_count": 0,
            "doc_type": "unknown",
        }

        # Add chunk position if provided
        if chunk_position:
            current, total = chunk_position
            structure["chunk_position"] = f"{current}/{total}"
            structure["chunk_index"] = current
            structure["total_chunks"] = total

        # Add section title if provided
        if section_title:
            structure["section_title"] = section_title

        # Extract headings based on format
        headings = []

        if doc_format in ("md", "markdown"):
            # Markdown headings: # H1, ## H2, ### H3
            heading_pattern = r"^(#{1,6})\s+(.+)$"
            for match in re.finditer(heading_pattern, text, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((level, title))

        elif doc_format in ("html", "htm"):
            # HTML headings: <h1>, <h2>, <h3>
            heading_pattern = r"<h([1-6])(?:[^>]*)>(.*?)</h\1>"
            for match in re.finditer(heading_pattern, text, re.IGNORECASE | re.DOTALL):
                level = int(match.group(1))
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                headings.append((level, title))

        else:
            # Generic: Look for lines that are all caps or followed by underline
            lines = text.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # All caps line (likely a heading)
                if len(line) > 3 and line.isupper() and not line.endswith(":"):
                    headings.append((1, line))

                # Line followed by === or --- (Markdown alternative heading)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r"^=+$", next_line):
                        headings.append((1, line))
                    elif re.match(r"^-+$", next_line):
                        headings.append((2, line))

        # Update structure metadata
        if headings:
            structure["has_headings"] = True
            structure["heading_count"] = len(headings)
            structure["headings"] = [{"level": lvl, "title": title} for lvl, title in headings[:5]]  # Limit to 5

            # Use first heading as section title if not provided
            if not section_title and headings:
                structure["section_title"] = headings[0][1]

        # Classify document type
        structure["doc_type"] = self._classify_document_type(text, doc_format)

        return structure

    def extract_semantic_metadata(self, text: str) -> Dict[str, Any]:
        """Extract semantic metadata from text.

        Detects:
        - Topics using TF-IDF
        - Keywords (most frequent important words)
        - Named entities (tools, technologies, people, places)

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with semantic metadata
        """
        semantic = {}

        # Extract keywords (frequent important words)
        keywords = self._extract_keywords(text, top_n=10)
        if keywords:
            semantic["keywords"] = keywords

        # Extract topics using TF-IDF (if enabled)
        if self.extract_topics and SKLEARN_AVAILABLE:
            topics = self._extract_topics_tfidf(text, top_n=5)
            if topics:
                semantic["topics"] = topics

        # Extract named entities (if enabled)
        if self.extract_entities:
            entities = self._extract_entities(text)
            if entities:
                semantic["entities"] = list(entities)[:15]  # Limit to 15
                semantic["entity_count"] = len(entities)

        return semantic

    def extract_technical_metadata(self, text: str, doc_format: str = "txt") -> Dict[str, Any]:
        """Extract technical content metadata.

        Detects:
        - Code blocks and programming language
        - Tables
        - Math equations
        - Function/class names (for code files)

        Args:
            text: Text content to analyze
            doc_format: Document format

        Returns:
            Dictionary with technical metadata
        """
        technical = {}

        # Detect code blocks (if enabled)
        if self.extract_code:
            code_blocks = self._detect_code_blocks(text)
            technical["has_code"] = len(code_blocks) > 0
            technical["code_block_count"] = len(code_blocks)

            # For code files, extract function/class names
            if doc_format in ("py", "js", "ts", "java", "cpp", "c", "rs", "go"):
                technical["programming_language"] = doc_format
                functions = self._extract_function_names(text, doc_format)
                classes = self._extract_class_names(text, doc_format)
                imports = self._extract_imports(text, doc_format)

                if functions:
                    technical["functions"] = functions[:10]  # Limit to 10
                    technical["function_count"] = len(functions)
                if classes:
                    technical["classes"] = classes[:10]  # Limit to 10
                    technical["class_count"] = len(classes)
                if imports:
                    technical["imports"] = imports[:10]  # Limit to 10
                    technical["import_count"] = len(imports)

        # Detect tables (if enabled)
        if self.extract_tables:
            table_count = self._count_tables(text)
            technical["has_tables"] = table_count > 0
            technical["table_count"] = table_count

        # Detect equations
        equation_count = self._count_equations(text)
        technical["has_equations"] = equation_count > 0
        technical["equation_count"] = equation_count

        return technical

    def extract_quality_signals(self, text: str) -> Dict[str, Any]:
        """Extract quality signals from text.

        Computes:
        - Word count
        - Sentence count
        - Average sentence length
        - Reading level (simple heuristic)

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with quality signals
        """
        quality = {}

        # Clean text for analysis
        clean_text = re.sub(r"\s+", " ", text).strip()

        # Word count
        words = clean_text.split()
        quality["word_count"] = len(words)

        # Sentence count
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(clean_text)
            except (LookupError, Exception):
                # Fallback if NLTK fails (e.g., missing punkt tokenizer)
                sentences = re.split(r"[.!?]+", clean_text)
                sentences = [s.strip() for s in sentences if s.strip()]
        else:
            # Fallback: Simple sentence splitting
            sentences = re.split(r"[.!?]+", clean_text)
            sentences = [s.strip() for s in sentences if s.strip()]

        quality["sentence_count"] = len(sentences)

        # Average sentence length
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
            quality["avg_sentence_length"] = round(avg_sentence_length, 1)

            # Reading level (simple heuristic based on avg sentence length)
            # < 10 words: very easy
            # 10-15: easy
            # 15-20: moderate
            # 20-25: difficult
            # > 25: very difficult
            if avg_sentence_length < 10:
                quality["reading_level"] = "very_easy"
            elif avg_sentence_length < 15:
                quality["reading_level"] = "easy"
            elif avg_sentence_length < 20:
                quality["reading_level"] = "moderate"
            elif avg_sentence_length < 25:
                quality["reading_level"] = "difficult"
            else:
                quality["reading_level"] = "very_difficult"

        # Character count
        quality["char_count"] = len(text)

        return quality

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _classify_document_type(self, text: str, doc_format: str) -> str:
        """Classify document type based on content patterns."""
        text_lower = text.lower()

        # Code files are always "code"
        if doc_format in ("py", "js", "ts", "java", "cpp", "c", "rs", "go"):
            return "code"

        # Count pattern matches for each type
        type_scores = {}
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower, re.IGNORECASE))
            if score > 0:
                type_scores[doc_type] = score

        # Return type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)

        return "general"

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords using word frequency."""
        # Tokenize
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text.lower())
                # Get stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                except (LookupError, OSError):
                    # Stopwords not downloaded
                    stop_words = set()
            except (LookupError, Exception):
                # Fallback if tokenization fails
                words = re.findall(r'\b[a-z]{3,}\b', text.lower())
                stop_words = set()
        else:
            # Fallback: Simple tokenization
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            stop_words = set()

        # Filter: only alphanumeric words, not stopwords, length > 2
        words = [
            w for w in words
            if w.isalnum() and len(w) > 2 and w not in stop_words
        ]

        # Count frequencies
        word_counts = Counter(words)

        # Return top N
        return [word for word, count in word_counts.most_common(top_n)]

    def _extract_topics_tfidf(self, text: str, top_n: int = 5) -> List[str]:
        """Extract topics using TF-IDF."""
        if not SKLEARN_AVAILABLE:
            return []

        try:
            # Split into sentences for TF-IDF
            if NLTK_AVAILABLE:
                try:
                    sentences = sent_tokenize(text)
                except (LookupError, Exception):
                    # Fallback if NLTK tokenization fails
                    sentences = re.split(r"[.!?]+", text)
                    sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = re.split(r"[.!?]+", text)
                sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) < 2:
                return []

            # Compute TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words='english',
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Get feature names and average scores
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1

            # Get top N features
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            topics = [feature_names[i] for i in top_indices]

            return topics

        except Exception as e:
            log.debug(f"TF-IDF topic extraction failed: {e}")
            return []

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities (tools, technologies)."""
        entities = set()
        text_lower = text.lower()

        # Check for known technical entities
        for entity in self.TECH_ENTITIES:
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{re.escape(entity)}\b', text_lower):
                entities.add(entity)

        # Extract capitalized words (potential proper nouns)
        # Pattern: Capitalized word not at start of sentence
        proper_nouns = re.findall(r'(?<!^)(?<!\. )\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)

        # Filter: only keep if appears multiple times (likely important)
        proper_noun_counts = Counter(proper_nouns)
        for noun, count in proper_noun_counts.items():
            if count >= 2:
                entities.add(noun.lower())

        return entities

    def _detect_code_blocks(self, text: str) -> List[str]:
        """Detect code blocks in text."""
        code_blocks = []

        for pattern in self.CODE_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            code_blocks.extend(matches)

        return code_blocks

    def _count_tables(self, text: str) -> int:
        """Count tables in text."""
        count = 0

        for pattern in self.TABLE_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            count += len(matches)

        return count

    def _count_equations(self, text: str) -> int:
        """Count math equations in text."""
        count = 0

        for pattern in self.EQUATION_PATTERNS:
            matches = re.findall(pattern, text)
            count += len(matches)

        return count

    def _extract_function_names(self, text: str, language: str) -> List[str]:
        """Extract function names from code."""
        functions = []

        if language == "py":
            # Python: def function_name(
            pattern = r'def\s+(\w+)\s*\('
        elif language in ("js", "ts"):
            # JavaScript/TypeScript: function name( or name = function( or name: function(
            pattern = r'(?:function\s+(\w+)|(\w+)\s*=\s*function|(\w+)\s*:\s*function)\s*\('
        elif language in ("java", "cpp", "c"):
            # Java/C++: type functionName(
            pattern = r'\b(?:public|private|protected|static|void|int|double|float|bool|string)\s+(\w+)\s*\('
        elif language == "go":
            # Go: func functionName(
            pattern = r'func\s+(\w+)\s*\('
        elif language == "rs":
            # Rust: fn function_name(
            pattern = r'fn\s+(\w+)\s*\('
        else:
            return []

        matches = re.findall(pattern, text)

        # Flatten (for patterns with multiple groups)
        for match in matches:
            if isinstance(match, tuple):
                name = next((m for m in match if m), None)
                if name:
                    functions.append(name)
            else:
                functions.append(match)

        return list(set(functions))  # Remove duplicates

    def _extract_class_names(self, text: str, language: str) -> List[str]:
        """Extract class names from code."""
        classes = []

        if language == "py":
            # Python: class ClassName
            pattern = r'class\s+(\w+)'
        elif language in ("js", "ts"):
            # JavaScript/TypeScript: class ClassName
            pattern = r'class\s+(\w+)'
        elif language in ("java", "cpp", "c"):
            # Java/C++: class ClassName
            pattern = r'\bclass\s+(\w+)'
        elif language == "go":
            # Go: type StructName struct
            pattern = r'type\s+(\w+)\s+struct'
        elif language == "rs":
            # Rust: struct StructName
            pattern = r'struct\s+(\w+)'
        else:
            return []

        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates

    def _extract_imports(self, text: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []

        if language == "py":
            # Python: import module or from module import
            pattern = r'(?:import|from)\s+([\w.]+)'
        elif language in ("js", "ts"):
            # JavaScript/TypeScript: import ... from 'module'
            pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        elif language == "java":
            # Java: import package.Class
            pattern = r'import\s+([\w.]+)'
        elif language in ("cpp", "c"):
            # C/C++: #include <header> or "header"
            pattern = r'#include\s+[<"]([^>"]+)[>"]'
        elif language == "go":
            # Go: import "package"
            pattern = r'import\s+[\'"]([^\'"]+)[\'"]'
        elif language == "rs":
            # Rust: use crate::module
            pattern = r'use\s+([\w:]+)'
        else:
            return []

        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates


class MetadataExtractor:
    """
    Extract enhanced metadata from text chunks.

    Provides structured information extraction for:
    - Dates and timestamps
    - Named entities (people, organizations, locations)
    - Content type classification
    - Quality metrics (length, complexity)
    - Structural elements (headings, lists, code blocks)
    """

    def __init__(self):
        """Initialize metadata extractor with pattern matchers."""
        # Date patterns (various formats)
        self.date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # 2024-01-15
            r'\b\d{2}/\d{2}/\d{4}\b',  # 01/15/2024
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # January 15, 2024
        ]

        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        # URL pattern
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        # Code block patterns
        self.code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`]+`',  # Inline code
            r'^\s*(?:def|class|function|var|let|const|import|from)\s',  # Code keywords
        ]

        # Common participant/speaker patterns for chat logs
        self.participant_pattern = r'^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*[:\-]'

    def extract(
        self,
        text: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract enhanced metadata from text.

        Args:
            text: Text content to analyze
            base_metadata: Existing metadata to extend (optional)

        Returns:
            Dict with extracted metadata fields
        """
        metadata = base_metadata.copy() if base_metadata else {}

        # Basic text statistics
        metadata['_char_count'] = len(text)
        metadata['_word_count'] = len(text.split())
        metadata['_line_count'] = len(text.split('\n'))

        # Extract dates
        dates = self._extract_dates(text)
        if dates:
            metadata['_dates'] = dates
            metadata['_date_count'] = len(dates)
            # Use first date as primary
            metadata['_primary_date'] = dates[0]

        # Extract emails
        emails = re.findall(self.email_pattern, text)
        if emails:
            metadata['_emails'] = list(set(emails))
            metadata['_email_count'] = len(emails)

        # Extract URLs
        urls = re.findall(self.url_pattern, text)
        if urls:
            metadata['_urls'] = list(set(urls))[:5]  # Limit to 5
            metadata['_url_count'] = len(urls)

        # Detect content type
        content_type = self._classify_content(text)
        if content_type:
            metadata['_content_type'] = content_type

        # Extract participants (for chat logs, transcripts)
        participants = self._extract_participants(text)
        if participants:
            metadata['_participants'] = list(participants.keys())
            metadata['_participant_count'] = len(participants)
            metadata['_participant_messages'] = participants

        # Detect code content
        has_code = any(re.search(pattern, text, re.MULTILINE) for pattern in self.code_patterns)
        if has_code:
            metadata['_has_code'] = True

        # Extract key phrases (simple version - just capitalized phrases)
        key_phrases = self._extract_key_phrases(text)
        if key_phrases:
            metadata['_key_phrases'] = key_phrases[:10]  # Top 10

        # Quality metrics
        metadata['_avg_word_length'] = sum(len(w) for w in text.split()) / max(1, len(text.split()))
        metadata['_unique_word_ratio'] = len(set(text.lower().split())) / max(1, len(text.split()))

        return metadata

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using multiple patterns"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        return list(dict.fromkeys(dates))  # Deduplicate while preserving order

    def _classify_content(self, text: str) -> Optional[str]:
        """
        Classify content type based on patterns.

        Returns:
            Content type: code, chat, email, documentation, article, etc.
        """
        text_lower = text.lower()

        # Code detection
        code_indicators = ['def ', 'class ', 'function ', 'import ', 'const ', 'var ']
        if any(indicator in text_lower for indicator in code_indicators):
            return 'code'

        # Chat/conversation detection
        if re.search(self.participant_pattern, text, re.MULTILINE):
            return 'chat'

        # Email detection
        if re.search(r'(from|to|subject|date):', text_lower, re.MULTILINE):
            return 'email'

        # Documentation detection (has headings and structured content)
        if re.search(r'^#+\s+\w+', text, re.MULTILINE):
            return 'documentation'

        # Default to article
        word_count = len(text.split())
        if word_count > 50:
            return 'article'

        return None

    def _extract_participants(self, text: str) -> Dict[str, int]:
        """
        Extract participants/speakers from chat logs or transcripts.

        Returns:
            Dict mapping participant names to message counts
        """
        participants = Counter()

        # Look for "Name:" or "Name -" patterns at start of lines
        for line in text.split('\n'):
            match = re.match(self.participant_pattern, line)
            if match:
                name = match.group(1)
                participants[name] += 1

        return dict(participants) if participants else {}

    def _extract_key_phrases(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract key phrases (simple capitalized phrases).

        This is a basic implementation. For production, consider using NER or TF-IDF.

        Args:
            text: Text to analyze
            min_length: Minimum number of words in phrase

        Returns:
            List of key phrases
        """
        # Find sequences of capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        phrases = re.findall(pattern, text)

        # Count occurrences
        phrase_counts = Counter(phrases)

        # Return phrases that appear multiple times or are long
        key_phrases = [
            phrase for phrase, count in phrase_counts.items()
            if count > 1 or len(phrase.split()) >= min_length
        ]

        return list(dict.fromkeys(key_phrases))  # Deduplicate, preserve order


def enhance_node_metadata(
    text: str,
    metadata: Dict[str, Any],
    extractor: Optional[DocumentMetadataExtractor] = None,
) -> Dict[str, Any]:
    """Enhance TextNode metadata with extracted metadata.

    This is a convenience function for integrating with build_nodes().

    Args:
        text: Text content of the node
        metadata: Existing metadata dict
        extractor: Optional DocumentMetadataExtractor instance

    Returns:
        Enhanced metadata dict

    Examples:
        >>> from utils.metadata_extractor import enhance_node_metadata
        >>> metadata = {"source": "doc.pdf", "format": "pdf"}
        >>> enhanced = enhance_node_metadata(text, metadata)
        >>> print(enhanced.get("struct_doc_type"))
        'research_paper'
    """
    if extractor is None:
        extractor = DocumentMetadataExtractor()

    if not extractor.enabled:
        return metadata

    # Extract metadata
    doc_format = metadata.get("format", "txt")
    doc_metadata = extractor.extract_all_metadata(text, doc_format=doc_format)

    # Merge with existing metadata
    enhanced = metadata.copy()
    enhanced.update(doc_metadata.to_dict())

    return enhanced


def main():
    """Test metadata extraction with comprehensive examples."""
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Test metadata extraction")
    parser.add_argument("--text", default=None, help="Text to analyze")
    parser.add_argument("--enhanced", action="store_true", help="Test enhanced metadata extractor")
    args = parser.parse_args()

    # Sample texts for testing
    sample_texts = {
        "chat": """
    Elena: Hey! I'm planning a trip to Morocco in June 2024.
    Alex: That sounds amazing! I went to Marrakech last year.
    Elena: Did you visit the Atlas Mountains?
    Alex: Yes! Email me at alex@example.com and I'll send you my itinerary.
    The website https://example.com/morocco has great info.

    Here's a Python snippet I used:
    ```python
    def plan_trip(destination):
        return f"Planning trip to {destination}"
    ```
    """,
        "tutorial": """
# Python Tutorial: Getting Started

## Introduction

Welcome to this Python tutorial. In this guide, you'll learn the basics of Python programming
using popular frameworks like Django, Flask, and FastAPI.

## Step 1: Installation

First, install Python from python.org. Make sure to add it to your PATH.

## Step 2: Your First Program

Create a file called `hello.py`:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("World")
```

This is a simple example of a Python function.
""",
        "research": """
Abstract: This paper presents a novel approach to machine learning optimization using PyTorch.

1. Introduction

Recent advances in deep learning have shown promising results. We propose a new method
for improving neural network training efficiency using TensorFlow and Keras.

2. Methodology

Our approach uses adaptive learning rates with the Adam optimizer. The key innovation is:

$$\\alpha_t = \\alpha_0 \\cdot \\sqrt{1 - \\beta_2^t} / (1 - \\beta_1^t)$$

3. Results

Experiments on ImageNet and CIFAR-10 datasets demonstrate 15% improvement over baseline.

| Dataset | Accuracy | Training Time |
|---------|----------|---------------|
| CIFAR-10| 95.2%    | 2.5 hours    |
| ImageNet| 87.3%    | 12 hours     |

4. Conclusion

We have shown that our method significantly improves training efficiency.

References:
[1] Smith et al., Neural Network Optimization, 2023
""",
        "code": """
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []

    def forward(self, X):
        return self.predict(X)

    def backward(self, X, y):
        self.update_weights(X, y)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

def main():
    model = NeuralNetwork([784, 128, 10])
    print("Model initialized")

if __name__ == "__main__":
    main()
""",
    }

    # Test enhanced extractor or basic extractor
    if args.enhanced:
        print("="*70)
        print("Enhanced Metadata Extraction Test (DocumentMetadataExtractor)")
        print("="*70)

        extractor = DocumentMetadataExtractor()

        for doc_type, text in sample_texts.items():
            print(f"\n{'='*70}")
            print(f"Testing: {doc_type.upper()}")
            print(f"{'='*70}\n")

            # Determine format
            doc_format = "md" if doc_type in ("tutorial", "research") else "py" if doc_type == "code" else "txt"

            # Extract metadata
            metadata = extractor.extract_all_metadata(text, doc_format=doc_format)

            # Print results
            print("Structure Metadata:")
            for k, v in metadata.structure.items():
                if k != "headings":  # Skip headings list for brevity
                    print(f"  {k}: {v}")

            print("\nSemantic Metadata:")
            for k, v in metadata.semantic.items():
                if isinstance(v, list):
                    print(f"  {k}: {', '.join(map(str, v[:5]))}")  # First 5 items
                else:
                    print(f"  {k}: {v}")

            print("\nTechnical Metadata:")
            for k, v in metadata.technical.items():
                if isinstance(v, list):
                    print(f"  {k}: {', '.join(map(str, v[:5]))}")  # First 5 items
                else:
                    print(f"  {k}: {v}")

            print("\nQuality Metadata:")
            for k, v in metadata.quality.items():
                print(f"  {k}: {v}")

        print(f"\n{'='*70}")
        print("Enhanced metadata extraction test complete!")
        print(f"{'='*70}\n")

    else:
        # Test basic extractor (backward compatibility)
        test_text = args.text or sample_texts["chat"]

        print("="*70)
        print("Basic Metadata Extraction Test (MetadataExtractor)")
        print("="*70)

        extractor = MetadataExtractor()

        print(f"\nAnalyzing text ({len(test_text)} characters)...")
        print(f"\nText preview:")
        print(f"{test_text[:200]}...")

        metadata = extractor.extract(test_text)

        print(f"\n\nExtracted Metadata:")
        print("-" * 70)
        for key, value in sorted(metadata.items()):
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")

        print("\n" + "="*70)
        print("Basic metadata extraction test complete!")
        print("="*70)

        print("\nTip: Run with --enhanced flag to test DocumentMetadataExtractor")


if __name__ == "__main__":
    main()
