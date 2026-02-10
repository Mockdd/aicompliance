"""
Hierarchical Chunking Script for Legal PDFs
Processes legal PDFs by splitting on Articles and applying size-based chunking when needed.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_as_single_text(pdf_path: Path) -> str:
    """
    Load a PDF and merge all pages into a single text string.
    This prevents Articles from being cut in half across page boundaries.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complete text content of the PDF as a single string
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    return full_text


def split_by_articles(text: str) -> List[tuple]:
    """
    Pass 1: Split text by Articles using regex.
    Looks for 'Article' followed by a number at the start of a line.
    
    Args:
        text: Full text content of the PDF
        
    Returns:
        List of tuples: (article_text, article_id)
    """
    # Pattern: Article followed by a number at the start of a line
    # Matches: "\nArticle 1", "\n  Article 2", "^Article 1" (start of text), etc.
    # Does NOT match inline references like "as seen in Article 5"
    # The pattern looks for (start of text OR newline) + optional whitespace + "Article" + number
    pattern = r'(?:^|\n)\s*(Article\s+\d+)'
    
    # Find all Article matches with their positions
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
    
    if not matches:
        # If no Articles found, return the entire text as a single chunk
        return [("", text)]
    
    article_chunks = []
    
    # Extract text before first Article (preamble)
    first_match_start = matches[0].start()
    if first_match_start > 0:
        preamble = text[:first_match_start].strip()
        if preamble:
            article_chunks.append(("", preamble))
    
    # Extract each Article
    for i, match in enumerate(matches):
        article_id = match.group(1)  # e.g., "Article 5"
        start_pos = match.start()
        
        # End position is the start of the next Article, or end of text
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        article_text = text[start_pos:end_pos].strip()
        article_chunks.append((article_id, article_text))
    
    return article_chunks


def chunk_article(article_text: str, article_id: str, 
                  max_chunk_size: int = 2000, 
                  chunk_size: int = 1000, 
                  chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Pass 2: Apply size-based chunking to Articles if they exceed the threshold.
    
    Args:
        article_text: Text content of a single Article
        article_id: Article identifier (e.g., "Article 5")
        max_chunk_size: Maximum size before splitting (default: 2000)
        chunk_size: Target chunk size for RecursiveCharacterTextSplitter (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        
    Returns:
        List of chunk dictionaries with 'page_content' and 'metadata'
    """
    chunks = []
    
    # If Article is small enough, keep as single chunk
    if len(article_text) <= max_chunk_size:
        chunks.append({
            'page_content': article_text,
            'metadata': {'article_id': article_id if article_id else 'Preamble'}
        })
    else:
        # Apply RecursiveCharacterTextSplitter for large Articles
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        sub_chunks = text_splitter.split_text(article_text)
        
        for sub_chunk in sub_chunks:
            chunks.append({
                'page_content': sub_chunk,
                'metadata': {'article_id': article_id if article_id else 'Preamble'}
            })
    
    return chunks


def process_pdf(pdf_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single PDF file through the hierarchical chunking pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        verbose: If True, print detailed progress messages
        
    Returns:
        List of all chunks with their metadata
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")
    
    # Step 1: Load PDF and merge all pages
    if verbose:
        print("Loading PDF and merging pages...")
    full_text = load_pdf_as_single_text(pdf_path)
    if verbose:
        print(f"Total text length: {len(full_text)} characters")
    
    # Step 2: Pass 1 - Split by Articles
    if verbose:
        print("\nPass 1: Splitting by Articles...")
    article_chunks = split_by_articles(full_text)
    if verbose:
        print(f"Found {len(article_chunks)} Article sections")
    
    # Step 3: Pass 2 - Apply size-based chunking
    if verbose:
        print("\nPass 2: Applying size-based chunking...")
    all_chunks = []
    
    for article_id, article_text in article_chunks:
        chunks = chunk_article(article_text, article_id)
        all_chunks.extend(chunks)
    
    return all_chunks


def print_summary(filename: str, chunks: List[Dict[str, Any]], verbose: bool = True):
    """
    Print summary statistics and sample chunks.
    
    Args:
        filename: Name of the processed file
        chunks: List of chunk dictionaries
        verbose: If True, print detailed sample chunks
    """
    # Count unique Articles
    article_ids = set()
    for chunk in chunks:
        article_id = chunk['metadata'].get('article_id', 'Unknown')
        article_ids.add(article_id)
    
    num_articles = len(article_ids)
    num_chunks = len(chunks)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Found {num_articles} Articles in {filename}.")
    print(f"Created {num_chunks} total chunks.")
    print(f"{'='*60}")
    
    # Print sample chunks (limited output to avoid overwhelming IDE)
    if verbose and num_chunks > 0:
        print(f"\n{'='*60}")
        print(f"SAMPLE CHUNKS (showing first 3)")
        print(f"{'='*60}")
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Article ID: {chunk['metadata'].get('article_id', 'N/A')}")
            print(f"Content Preview ({len(chunk['page_content'])} chars):")
            # Limit preview to 150 chars to reduce output size
            preview = chunk['page_content'][:150] + "..." if len(chunk['page_content']) > 150 else chunk['page_content']
            print(f"{preview}")
            print(f"Metadata: {chunk['metadata']}")


def save_chunks_to_file(chunks: List[Dict[str, Any]], output_path: Path):
    """
    Save chunks to a JSON file for later use.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the JSON file
    """
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to: {output_path}")


def main():
    """
    Main function to process all PDFs in the data/raw directory.
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
    
    # Find all PDF files
    pdf_files = list(raw_data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {raw_data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            # Process with minimal verbosity to avoid overwhelming IDE
            chunks = process_pdf(pdf_path, verbose=False)
            
            # Print concise summary
            print_summary(pdf_path.name, chunks, verbose=True)
            
            # Save chunks to JSON file
            output_file = output_dir / f"{pdf_path.stem}_chunks.json"
            save_chunks_to_file(chunks, output_file)
            
        except Exception as e:
            print(f"\nError processing {pdf_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
