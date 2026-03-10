# Memo Inferencer

AI-powered pitch deck analysis for InSync startup memo autofill.

This tool uses **Unstructured.io's VLM (Vision Language Model) strategy** to
understand the *visual layout* of each slide—capturing titles, tables, charts,
and context that traditional text extraction misses.

## Pipeline

### Step 1: Deck Distillation ✅

**Input:** Firebase Storage URL pointing to a pitch deck PDF

**Process:**
1. Download the PDF from the publicly readable URL
2. Send it to Unstructured.io's Partition API with `Strategy.VLM`
   - Uses computer vision (GPT-4o) to understand slide layout
   - `split_pdf_page=True` keeps each slide as its own context unit
3. Group extracted elements by slide number
4. Each element is tagged with its semantic type:
   - `Title` — slide headings
   - `NarrativeText` — body paragraphs
   - `Table` — structured data (with HTML rendering)
   - `ListItem` — bullet points
   - `Image` / `FigureCaption` — visual elements
5. Render everything into a structured `.md` Markdown document

**Output:** A clean Markdown file organized slide-by-slide, e.g.:
```markdown
# Slide 1
## InSync: The Future of AI Matchmaking
InSync connects VC firms with startups using AI-driven analysis...

# Slide 2
## The Problem
- Current networking tools are broken and inefficient...
- Startups waste 40% of fundraising time on mismatched VCs

# Slide 5
## Traction
**[Table]**
| Metric | Q1 | Q2 | Q3 |
| MRR    | $10K | $25K | $50K |
```

### Step 2: LLM Memo Inference (Coming Soon)

Will take the distilled Markdown and pass it to an LLM with a strict system
prompt to extract structured memo data matching the `AutofillResponse` schema.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Unstructured.io API key
```

To get an API key:
1. Sign up at https://unstructured.io/?modal=try-for-free
2. Go to https://platform.unstructured.io → **API Keys** → **Generate API Key**

### 4. Run the distiller

```bash
# Basic usage
python run.py "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/deck.pdf?alt=media"

# With options
python run.py "https://your-url.com/deck.pdf" --strategy vlm --output-dir ./output --print
```

---

## Project Structure

```
memo-inferencer/
├── run.py              # Main entry point / CLI
├── deck_distiller.py   # Step 1: PDF → Markdown via Unstructured.io
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore
├── README.md
└── output/             # Generated markdown files (gitignored)
```

## Why Unstructured.io over basic PDF tools?

| Feature | pdfplumber (ocr-tool) | Unstructured.io VLM |
|---------|----------------------|---------------------|
| Text extraction | ✅ Basic | ✅ AI-powered |
| Layout understanding | ❌ | ✅ Vision models |
| Table detection | ⚠️ Limited | ✅ Structured HTML |
| Semantic labeling | ❌ | ✅ Title, List, Table, etc. |
| Chart/image awareness | ❌ | ✅ Figure captions |
| Slide context isolation | ⚠️ Manual | ✅ `split_pdf_page` |
