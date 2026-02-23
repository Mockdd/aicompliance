# Fine-Tuning Corpus

Unified corpus for continued pre-training on AI compliance (AI Act, Korea AI Law) and guidelines.

## Format

See [FORMAT_SPEC.md](FORMAT_SPEC.md) for the full specification.

Each line in `corpus.jsonl` is a JSON object:
```json
{"meta": {"regulation": "AIAct", "section_type": "article", "article_ref": "Article 71", ...}, "text": "..."}
```

## Data Sources

1. **Relations JSON** (`AIAct_relations_ready.json`, `AICoreLaw_relations_ready.json`)
   - Only text from relations labeled with nodes `(Sanction)` and `(Requirement)`
   - Explicit `section_type`: article, addenda, recitals, annex
   - Explicit `regulation`: AIAct or KRAILaw

2. **Guidelines** (`data/guidelines/*.pdf`)
   - Domain sections (Healthcare, Education, Employment, etc.)
   - Article-specific documents
   - Place PDFs in `data/guidelines/`

## Usage

```bash
# Process relations only
python -m src.prepare_finetune_data --relations-only

# Process guidelines only
python -m src.prepare_finetune_data --guidelines-only

# Process both (default)
python -m src.prepare_finetune_data

# Custom output path
python -m src.prepare_finetune_data -o data/finetune/my_corpus.jsonl
```

## Output

- **corpus.jsonl**: One JSON object per line, ready for causal LM fine-tuning.
