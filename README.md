# KUBIG AI Compliance

Legal document processing pipeline for Neo4j AuraDB ingestion.

## Overview

This project processes legal documents (EU AI Act and AI Core Law) and converts them into structured JSON format suitable for Neo4j graph database import.

## Features

- **HTML Parsing**: Processes `AIAct.html` using BeautifulSoup to extract Recitals, Articles, and Annexes
- **PDF Parsing**: Processes `AICoreLaw.pdf` using pypdf with Korean character removal
- **Graph Structure**: Outputs parent-child relationships (Articles/Rationales/Annexes → Chunks)
- **Vector Chunking**: Splits documents into chunks (1000 chars, 200 overlap) for embedding

## Project Structure

```
KUBIG_aicompliance/
├── src/
│   ├── ingest_legal_optimized.py  # Main processing script
│   └── hierarchical_chunking.py    # Chunking utilities
├── data/
│   ├── raw/                        # Source documents (gitignored)
│   └── processed/                  # Output JSON files (gitignored)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your source documents in `data/raw/`:
   - `AIAct.html`
   - `AICoreLaw.pdf`

3. Run the processing script:
```bash
python src/ingest_legal_optimized.py
```

4. Output files will be generated in `data/processed/`:
   - `AIAct_graph_chunk.json`
   - `AICoreLaw_graph_chunk.json`

## Dependencies

- langchain >= 0.1.0
- langchain-text-splitters >= 1.0.0
- pypdf >= 3.0.0
- beautifulsoup4 >= 4.12.0

## Notes

- API keys and sensitive data are excluded via `.gitignore`
- Large data files are excluded from version control
- Processed outputs should be regenerated from source documents
