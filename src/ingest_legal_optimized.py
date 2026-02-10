"""
Legal PDF Ingestion Script for Neo4j AuraDB
Processes legal PDFs into parent-child structure optimized for graph database.
Supports: Rationale (Recitals), Articles, and Annexes as parent nodes.
Saves separate JSON files for each PDF.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace but preserve structure.
    Also removes header/footer patterns.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove header/footer patterns (e.g., "EN\nOJ L, 12.7.2024\n44/144 ELI: ...")
    # Pattern: \nEN\nOJ L, 12.7.2024\n44/144 ELI: http://data.europa.eu/eli/reg/2024/1689/oj\n
    text = re.sub(r'\nEN\s*\nOJ\s+L[,\s]+\d+\.\d+\.\d+\s*\n\d+/\d+\s+ELI:.*?\n', '\n', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\nEN\s*\nOJ\s+L[,\s]+\d+\.\d+\.\d+.*?\n', '\n', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\n\d+/\d+\s+ELI:.*?\n', '\n', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize whitespace but keep newlines for structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    return text.strip()


def remove_korean_characters(text: str) -> str:
    """
    Remove all Korean characters (Hangul) from text.
    Uses Unicode range [\u3131-\uD79D] for Hangul.
    
    Args:
        text: Text containing Korean characters
        
    Returns:
        Text with Korean characters removed
    """
    text = re.sub(r'[\u3131-\uD79D]', '', text)
    return text


def load_pdf_with_page_ranges(pdf_path: Path, 
                              recitals_end: Optional[int] = None,
                              articles_end: Optional[int] = None) -> Tuple[str, str, str]:
    """
    Load PDF and split into three sections: Recitals, Articles, Annexes.
    
    Args:
        pdf_path: Path to the PDF file
        recitals_end: Page number where recitals end (exclusive, 1-indexed)
        articles_end: Page number where articles end (exclusive, 1-indexed)
        
    Returns:
        Tuple of (recitals_text, articles_text, annexes_text)
    """
    reader = PdfReader(pdf_path)
    recitals_text = ""
    articles_text = ""
    annexes_text = ""
    
    for i, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() + "\n"
        
        if recitals_end and i < recitals_end:
            recitals_text += page_text
        elif articles_end and recitals_end and recitals_end <= i < articles_end:
            articles_text += page_text
        elif articles_end and i >= articles_end:
            annexes_text += page_text
        elif recitals_end and i >= recitals_end:
            articles_text += page_text
        else:
            recitals_text += page_text
    
    return recitals_text, articles_text, annexes_text


def split_recitals(text: str) -> List[Dict[str, Any]]:
    """
    Split recitals text into individual Rationale nodes.
    Uses pattern ^(\\d+) to match recitals at start of line.
    
    Args:
        text: Recitals text
        
    Returns:
        List of rationale dictionaries
    """
    rationales = []
    # Pattern to match numbered recitals at start of line: ^(\d+)
    recital_pattern = re.compile(r'^\((\d+)\)', re.MULTILINE)
    
    # Find all recital matches
    matches = list(recital_pattern.finditer(text))
    
    if not matches:
        # If no numbered recitals found, treat entire text as one rationale
        if text.strip():
            rationales.append({
                'id': 'Recital 1',
                'full_text': text.strip()
            })
        return rationales
    
    # Extract each recital
    for i, match in enumerate(matches):
        recital_num = match.group(1)
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        recital_text = text[start_pos:end_pos].strip()
        if recital_text:
            rationales.append({
                'id': f'Recital {recital_num}',
                'full_text': recital_text
            })
    
    return rationales


def assign_chapter_to_article(article_num: int) -> str:
    """
    Assign CHAPTER to Article based on article number.
    Uses the provided mapping from the user.
    
    Args:
        article_num: Article number as integer
        
    Returns:
        CHAPTER identifier string
    """
    if 1 <= article_num <= 4:
        return "CHAPTER I"
    elif article_num == 5:
        return "CHAPTER II PROHIBITED AI PRACTICES"
    elif 6 <= article_num <= 49:
        return "CHAPTER III High-Risk AI Systems"
    elif article_num == 50:
        return "CHAPTER IV TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS"
    elif 51 <= article_num <= 56:
        return "CHAPTER V TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS"
    elif 57 <= article_num <= 63:
        return "CHAPTER VI MEASURES IN SUPPORT OF INNOVATION"
    elif 64 <= article_num <= 70:
        return "CHAPTER VII GOVERNANCE"
    elif article_num == 71:
        return "CHAPTER VIII EU DATABASE FOR HIGH-RISK AI SYSTEMS"
    elif 72 <= article_num <= 94:
        return "CHAPTER IX POST-MARKET MONITORING, INFORMATION SHARING AND MARKET SURVEILLANCE"
    elif 95 <= article_num <= 96:
        return "CHAPTER X DELEGATION OF POWER AND COMMITTEE PROCEDURE"
    elif 97 <= article_num <= 98:
        return "CHAPTER XI DELEGATION OF POWER AND COMMITTEE PROCEDURE"
    elif 99 <= article_num <= 101:
        return "CHAPTER XII PENALTIES"
    elif 102 <= article_num <= 113:
        return "CHAPTER XIII FINAL PROVISIONS"
    else:
        return None


def extract_articles_with_chapter_tracking(text: str) -> List[Dict[str, Any]]:
    """
    Extract Articles with CHAPTER tracking.
    Handles PDF spacing issues (e.g., "Ar ticle" -> "Article").
    
    Args:
        text: Articles text
        
    Returns:
        List of article dictionaries with chapter metadata
    """
    # Normalize spacing issues in text (e.g., "Ar ticle" -> "Article")
    # Handle various spacing patterns
    text = re.sub(r'A\s*r\s*t\s*i\s*c\s*l\s*e', 'Article', text, flags=re.IGNORECASE)
    text = re.sub(r'C\s*H\s*A\s*P\s*T\s*E\s*R', 'CHAPTER', text, flags=re.IGNORECASE)
    # Also handle simpler spacing: "Ar ticle" -> "Article"
    text = re.sub(r'A\s+r\s+t\s+i\s+c\s+l\s+e', 'Article', text, flags=re.IGNORECASE)
    text = re.sub(r'C\s+h\s+a\s+p\s+t\s+e\s+r', 'CHAPTER', text, flags=re.IGNORECASE)
    
    lines = text.split('\n')
    articles = []
    
    # State tracking
    current_chapter = None
    current_article_id = None
    current_article_lines = []
    
    # Patterns - match Article/CHAPTER even with spaces
    chapter_pattern = re.compile(r'^C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)', re.IGNORECASE)
    # Match "Article" with any spacing: "Ar ticle 2", "Article 2", etc.
    article_pattern = re.compile(r'^A\s*r\s*t\s*i\s*c\s*l\s*e\s+(\d+)', re.IGNORECASE)
    
    def save_current_article():
        """Helper to save current article before starting new one."""
        nonlocal current_article_id, current_article_lines, articles
        if current_article_id and current_article_lines:
            # Normalize the article text
            article_text = '\n'.join(current_article_lines).strip()
            # Clean up spacing in the text
            article_text = re.sub(r'A\s+r\s+t\s+i\s+c\s+l\s+e', 'Article', article_text, flags=re.IGNORECASE)
            
            # Assign chapter based on article number if not found in text
            article_num = int(current_article_id)
            assigned_chapter = assign_chapter_to_article(article_num)
            final_chapter = current_chapter if current_chapter else assigned_chapter
            
            articles.append({
                'id': f"Article {current_article_id}",
                'metadata': {
                    'chapter': final_chapter
                },
                'full_text': article_text
            })
            current_article_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for Chapter (handle spacing)
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            save_current_article()
            current_chapter = f"CHAPTER {chapter_match.group(1)}"
            continue
        
        # Check for Article (handle spacing)
        article_match = article_pattern.match(line_stripped)
        if article_match:
            save_current_article()
            current_article_id = article_match.group(1)
            # Normalize the line
            normalized_line = re.sub(r'A\s+r\s+t\s+i\s+c\s+l\s+e', 'Article', line_stripped, flags=re.IGNORECASE)
            current_article_lines = [normalized_line]
            continue
        
        # Accumulate text for current article
        if current_article_id:
            # Normalize spacing in accumulated lines too
            normalized_line = re.sub(r'A\s+r\s+t\s+i\s+c\s+l\s+e', 'Article', line, flags=re.IGNORECASE)
            current_article_lines.append(normalized_line)
    
    # Save last article
    save_current_article()
    
    return articles


def extract_annexes(text: str) -> List[Dict[str, Any]]:
    """
    Extract Annexes from text.
    Looks for headers like "ANNEX I", "ANNEX II", etc. (Roman numerals).
    
    Args:
        text: Annexes text
        
    Returns:
        List of annex dictionaries
    """
    annexes = []
    # Pattern to match "ANNEX" followed by Roman numerals at start of line
    annex_pattern = re.compile(r'^ANNEX\s+([IVX]+)', re.IGNORECASE | re.MULTILINE)
    
    # Find all annex matches
    matches = list(annex_pattern.finditer(text))
    
    if not matches:
        return annexes
    
    # Extract each annex
    for i, match in enumerate(matches):
        annex_num = match.group(1)
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        annex_text = text[start_pos:end_pos].strip()
        
        # Try to extract title (first substantial line after ANNEX header)
        title = None
        lines_after_header = annex_text.split('\n')[:10]
        for line in lines_after_header:
            line_clean = line.strip()
            if line_clean and not re.match(r'^ANNEX\s+[IVX]+', line_clean, re.IGNORECASE):
                # Extract potential title (first substantial line)
                if len(line_clean) > 10 and len(line_clean) < 200:
                    title = line_clean
                    break
        
        annexes.append({
            'id': f'ANNEX {annex_num}',
            'title': title or f'Annex {annex_num}',
            'full_text': annex_text
        })
    
    return annexes


def create_chunks_with_structure(parent_text: str, parent_id: str, 
                                chunk_size: int = 1000, 
                                chunk_overlap: int = 200,
                                custom_separators: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Create vector chunks from parent text, respecting paragraph structure.
    
    Args:
        parent_text: Full parent text
        parent_id: Parent identifier
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        custom_separators: Custom separators for splitting (for Articles)
        
    Returns:
        List of chunk dictionaries
    """
    if not parent_text or len(parent_text.strip()) == 0:
        return []
    
    # Clean parent_id for chunk_id generation
    clean_id = parent_id.replace(' ', '_').replace('ANNEX', 'annex').lower()
    
    # If parent is small enough, create single chunk
    if len(parent_text) <= chunk_size:
        return [{
            'chunk_id': f"{clean_id}_c1",
            'text': parent_text.strip()
        }]
    
    # Use custom separators for Articles to respect paragraph structure
    if custom_separators:
        separators = custom_separators
    else:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    # Use RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators
    )
    
    sub_chunks = text_splitter.split_text(parent_text)
    chunks = []
    
    for i, chunk_text in enumerate(sub_chunks, 1):
        chunks.append({
            'chunk_id': f"{clean_id}_c{i}",
            'text': chunk_text.strip()
        })
    
    return chunks


def process_aiact_html(html_path: Path) -> Dict[str, Any]:
    """
    Process AIAct.html with Recitals/Articles/Annexes separation.
    Uses BeautifulSoup to parse HTML structure and iterate through elements.
    
    Args:
        html_path: Path to AIAct.html
        
    Returns:
        Dictionary with source, rationale, articles, and annexes
    """
    # Load and parse HTML
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Find the anchor point: "HAVE ADOPTED THIS REGULATION"
    adoption_elem = None
    for p in soup.find_all('p', class_='oj-normal'):
        if p.get_text() and 'HAVE ADOPTED THIS REGULATION' in p.get_text():
            adoption_elem = p
            break
    
    # Collect all text elements in document order
    # Only process <p> elements to avoid duplication (divs contain p elements)
    all_elements = soup.find_all('p')
    
    # Track state
    found_adoption = False
    current_chapter = None
    current_article_id = None
    current_article_text = []
    current_recital_num = None
    current_recital_text = []
    current_annex_num = None
    current_annex_text = []
    
    rationales = []
    articles = []
    annexes = []
    
    # Helper function to get clean text from element
    def get_element_text(elem):
        if not elem:
            return ""
        text = elem.get_text(separator='\n', strip=True)
        return text.strip()
    
    # Iterate through elements in document order
    for elem in all_elements:
        text = get_element_text(elem)
        
        # Skip empty elements
        if not text:
            continue
        
        # Check if we've reached the adoption marker
        if adoption_elem and elem == adoption_elem:
            found_adoption = True
            # Save any pending recital
            if current_recital_num and current_recital_text:
                rationales.append({
                    'id': f'Recital {current_recital_num}',
                    'full_text': '\n'.join(current_recital_text).strip()
                })
                current_recital_text = []
            continue
        
        # Before adoption: Process as Recitals
        if not found_adoption:
            # Check for recital pattern: (N) at start
            recital_match = re.match(r'^\((\d+)\)', text)
            if recital_match:
                # Save previous recital if exists
                if current_recital_num and current_recital_text:
                    rationales.append({
                        'id': f'Recital {current_recital_num}',
                        'full_text': '\n'.join(current_recital_text).strip()
                    })
                current_recital_num = recital_match.group(1)
                current_recital_text = [text]
            elif current_recital_num and text:
                # Continue accumulating recital text
                current_recital_text.append(text)
        
        # After adoption: Process as Articles/Annexes
        else:
            # Skip if we're in an annex section
            if current_annex_num:
                # Check for next annex (with oj-doc-ti class) or article (with oj-ti-art class)
                if ('oj-doc-ti' in elem.get('class', []) and re.match(r'^ANNEX\s+([IVX]+)', text, re.I)) or \
                   ('oj-ti-art' in elem.get('class', []) and re.match(r'^Article\s+\d+', text, re.I)):
                    # Save current annex
                    if current_annex_text:
                        annexes.append({
                            'id': f'ANNEX {current_annex_num}',
                            'title': f'Annex {current_annex_num}',
                            'full_text': '\n'.join(current_annex_text).strip()
                        })
                        current_annex_text = []
                        current_annex_num = None
                else:
                    if text:
                        current_annex_text.append(text)
                    continue
            
            # Check for Annex - only match if element has class 'oj-doc-ti' (annex header)
            if 'oj-doc-ti' in elem.get('class', []):
                annex_match = re.match(r'^ANNEX\s+([IVX]+)', text, re.I)
                if annex_match:
                    # Save current article if exists
                    if current_article_id and current_article_text:
                        article_num = int(current_article_id)
                        assigned_chapter = assign_chapter_to_article(article_num)
                        final_chapter = current_chapter if current_chapter else assigned_chapter
                        articles.append({
                            'id': f"Article {current_article_id}",
                            'metadata': {'chapter': final_chapter},
                            'full_text': '\n'.join(current_article_text).strip()
                        })
                        current_article_text = []
                        current_article_id = None
                    
                    # Start new annex
                    current_annex_num = annex_match.group(1)
                    current_annex_text = [text]
                    continue
            
            # Skip chapter titles and section titles - these should not be accumulated into articles
            if 'oj-ti-section-2' in elem.get('class', []):
                continue
            
            # Check for Chapter - only match if element has class 'oj-ti-section-1' (chapter header)
            if 'oj-ti-section-1' in elem.get('class', []):
                chapter_match = re.match(r'^CHAPTER\s+([IVX]+|\d+)', text, re.I)
                if chapter_match:
                    # Save current article if exists
                    if current_article_id and current_article_text:
                        article_num = int(current_article_id)
                        assigned_chapter = assign_chapter_to_article(article_num)
                        final_chapter = current_chapter if current_chapter else assigned_chapter
                        articles.append({
                            'id': f"Article {current_article_id}",
                            'metadata': {'chapter': final_chapter},
                            'full_text': '\n'.join(current_article_text).strip()
                        })
                        current_article_text = []
                        current_article_id = None  # Clear article ID after saving
                    
                    # Extract chapter - try to get full text
                    chapter_val = chapter_match.group(1)
                    if chapter_val.isdigit():
                        current_chapter = f"CHAPTER {chapter_val}"
                    else:
                        # Try to get more context from the element
                        current_chapter = text.strip()
                    continue
            
            # Check for Article - only match if element has class 'oj-ti-art' (article header)
            # This prevents matching references like "Article 6(1)" in the middle of text
            if 'oj-ti-art' in elem.get('class', []):
                article_match = re.match(r'^Article\s+(\d+)', text, re.I)
                if article_match:
                    # Save previous article if exists
                    if current_article_id and current_article_text:
                        article_num = int(current_article_id)
                        assigned_chapter = assign_chapter_to_article(article_num)
                        final_chapter = current_chapter if current_chapter else assigned_chapter
                        articles.append({
                            'id': f"Article {current_article_id}",
                            'metadata': {'chapter': final_chapter},
                            'full_text': '\n'.join(current_article_text).strip()
                        })
                    
                    current_article_id = article_match.group(1)
                    current_article_text = [text]
                    continue
            
            # Accumulate text for current article
            # Only accumulate if we have an active article and the text is not empty
            # Also skip if this is a chapter/section title that somehow got through
            if current_article_id and text and 'oj-ti-section' not in ' '.join(elem.get('class', [])):
                current_article_text.append(text)
    
    # Save last recital
    if current_recital_num and current_recital_text:
        rationales.append({
            'id': f'Recital {current_recital_num}',
            'full_text': '\n'.join(current_recital_text).strip()
        })
    
    # Save last article
    if current_article_id and current_article_text:
        article_num = int(current_article_id)
        assigned_chapter = assign_chapter_to_article(article_num)
        final_chapter = current_chapter if current_chapter else assigned_chapter
        articles.append({
            'id': f"Article {current_article_id}",
            'metadata': {'chapter': final_chapter},
            'full_text': '\n'.join(current_article_text).strip()
        })
    
    # Save last annex
    if current_annex_num and current_annex_text:
        annexes.append({
            'id': f'ANNEX {current_annex_num}',
            'title': f'Annex {current_annex_num}',
            'full_text': '\n'.join(current_annex_text).strip()
        })
    
    # Clean and process rationales
    for rationale in rationales:
        rationale['full_text'] = clean_text(rationale['full_text'])
        rationale['chunks'] = create_chunks_with_structure(
            rationale['full_text'],
            rationale['id']
        )
    
    # Clean and process articles
    article_separators = ["\n\n", "\nArticle ", "\n", ". ", " "]
    for article in articles:
        article['full_text'] = clean_text(article['full_text'])
        article['chunks'] = create_chunks_with_structure(
            article['full_text'],
            article['id'],
            custom_separators=article_separators
        )
    
    # Clean and process annexes
    for annex in annexes:
        annex['full_text'] = clean_text(annex['full_text'])
        annex['chunks'] = create_chunks_with_structure(
            annex['full_text'],
            annex['id']
        )
    
    return {
        'source': 'EU AI Act',
        'rationale': rationales,
        'articles': articles,
        'annexes': annexes
    }


def process_aicorelaw_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Process AICoreLaw.pdf: Remove Korean characters, ignore ADDENDA, extract Articles with titles.
    
    Args:
        pdf_path: Path to AICoreLaw.pdf
        
    Returns:
        Dictionary with source, articles, empty rationale and annexes
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # Load pages but skip ADDENDA (pages 39-40, 0-indexed: 38-39)
    for i, page in enumerate(reader.pages):
        if i == 38 or i == 39:  # Skip ADDENDA pages
            continue
        full_text += page.extract_text() + "\n"
    
    # Remove Korean characters
    english_text = remove_korean_characters(full_text)
    
    # Clean text
    english_text = clean_text(english_text)
    
    # Normalize spacing issues
    english_text = re.sub(r'A\s*r\s*t\s*i\s*c\s*l\s*e', 'Article', english_text, flags=re.IGNORECASE)
    english_text = re.sub(r'C\s*H\s*A\s*P\s*T\s*E\s*R', 'CHAPTER', english_text, flags=re.IGNORECASE)
    
    # Track current chapter
    current_chapter = None
    lines = english_text.split('\n')
    articles = []
    current_article_id = None
    current_article_title = None
    current_article_lines = []
    
    chapter_pattern = re.compile(r'^CHAPTER\s+(\d+)', re.IGNORECASE)
    # Pattern for Article with optional title: Article \d+ (Title)
    article_pattern = re.compile(r'^Article\s+(\d+)(?:\s*\(([^)]+)\))?', re.IGNORECASE)
    
    def save_current_article():
        """Helper to save current article."""
        nonlocal current_article_id, current_article_title, current_article_lines, articles
        if current_article_id and current_article_lines:
            metadata = {'chapter': current_chapter}
            if current_article_title:
                metadata['title'] = current_article_title
            
            articles.append({
                'id': f"Article {current_article_id}",
                'metadata': metadata,
                'full_text': '\n'.join(current_article_lines).strip()
            })
            current_article_lines = []
            current_article_title = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for Chapter
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            save_current_article()
            current_chapter = f"CHAPTER {chapter_match.group(1)}"
            continue
        
        # Check for Article
        article_match = article_pattern.match(line_stripped)
        if article_match:
            save_current_article()
            current_article_id = article_match.group(1)
            current_article_title = article_match.group(2)  # May be None
            current_article_lines = [line_stripped]
            continue
        
        # Accumulate text for current article
        if current_article_id:
            current_article_lines.append(line)
    
    # Save last article
    save_current_article()
    
    # Create chunks for each article
    for article in articles:
        article['chunks'] = create_chunks_with_structure(
            article['full_text'], 
            article['id']
        )
    
    return {
        'source': 'Korea AI Law',
        'rationale': [],
        'articles': articles,
        'annexes': []
    }


def main():
    """
    Main function to process legal PDFs and create separate Neo4j-ready JSON files.
    """
    import json
    
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_data_dir.exists():
        print(f"Error: Directory {raw_data_dir} does not exist!")
        return
    
    # Process AIAct.html
    aiact_path = raw_data_dir / "AIAct.html"
    if aiact_path.exists():
        print(f"\nProcessing: {aiact_path.name}")
        try:
            aiact_data = process_aiact_html(aiact_path)
            
            # Count nodes
            num_rationale = len(aiact_data['rationale'])
            num_articles = len(aiact_data['articles'])
            num_annexes = len(aiact_data['annexes'])
            total_chunks = (
                sum(len(r['chunks']) for r in aiact_data['rationale']) +
                sum(len(a['chunks']) for a in aiact_data['articles']) +
                sum(len(an['chunks']) for an in aiact_data['annexes'])
            )
            
            print(f"Found {num_rationale} Rationale, {num_articles} Articles, {num_annexes} Annexes, and {total_chunks} Chunks.")
            
            # Save to separate file
            output_file = output_dir / "AIAct_graph_chunk.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aiact_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved to: {output_file}")
            
            # Print example Annex node
            if aiact_data['annexes']:
                sample_annex = aiact_data['annexes'][0]
                print(f"\n{'='*60}")
                print("EXAMPLE ANNEX NODE:")
                print(f"{'='*60}")
                print(json.dumps({
                    'id': sample_annex['id'],
                    'title': sample_annex.get('title', 'N/A'),
                    'full_text_preview': sample_annex['full_text'][:200] + "..." if len(sample_annex['full_text']) > 200 else sample_annex['full_text'],
                    'chunks_count': len(sample_annex['chunks']),
                    'first_chunk_id': sample_annex['chunks'][0]['chunk_id'] if sample_annex['chunks'] else None
                }, indent=2, ensure_ascii=False))
                
        except Exception as e:
            print(f"Error processing {aiact_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Process AICoreLaw.pdf
    aicorelaw_path = raw_data_dir / "AICoreLaw.pdf"
    if aicorelaw_path.exists():
        print(f"\nProcessing: {aicorelaw_path.name}")
        try:
            aicorelaw_data = process_aicorelaw_pdf(aicorelaw_path)
            
            # Count nodes
            num_rationale = len(aicorelaw_data['rationale'])
            num_articles = len(aicorelaw_data['articles'])
            num_annexes = len(aicorelaw_data['annexes'])
            total_chunks = (
                sum(len(r['chunks']) for r in aicorelaw_data['rationale']) +
                sum(len(a['chunks']) for a in aicorelaw_data['articles']) +
                sum(len(an['chunks']) for an in aicorelaw_data['annexes'])
            )
            
            print(f"Found {num_rationale} Rationale, {num_articles} Articles, {num_annexes} Annexes, and {total_chunks} Chunks.")
            
            # Save to separate file
            output_file = output_dir / "AICoreLaw_graph_chunk.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aicorelaw_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {aicorelaw_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
