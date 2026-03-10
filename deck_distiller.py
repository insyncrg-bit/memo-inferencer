"""
Deck Distillation – Step 1 of the Memo Inferencer pipeline.

Takes a pitch deck PDF (via a publicly readable Firebase bucket URL) and uses
Unstructured.io's Partition API with the VLM strategy to extract structured,
labelled elements from every slide.  The output is a clean Markdown (.md)
document organized slide-by-slide, with each element tagged by its semantic
type (Title, NarrativeText, Table, ListItem, etc.).

Why Unstructured instead of a plain PDF reader?
    • VLM strategy uses vision-language models (e.g. GPT-4o) to *look* at the
      page layout – it understands floating text boxes, sidebars, charts, etc.
    • split_pdf_page keeps every slide as its own context unit so content from
      "The Problem" slide never bleeds into "The Solution" slide.
    • Each element comes back with a semantic type label – Title, Table,
      NarrativeText, ListItem – giving downstream LLM prompts richer context.

Usage:
    from deck_distiller import distill_deck
    markdown = await distill_deck("https://firebasestorage.googleapis.com/...")
"""
from __future__ import annotations

import os
import re
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

import unstructured_client
from unstructured_client.models import operations, shared

load_dotenv()

# ─── Constants ─────────────────────────────────────────────────────────────────

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")

# VLM model configuration – Unstructured uses this for its vision strategy.
# Supported providers: openai, anthropic, bedrock, vertexai
VLM_MODEL_PROVIDER = "openai"
VLM_MODEL = "gpt-4o"

# Concurrency settings for split-PDF parallel processing
SPLIT_PDF_CONCURRENCY = 15


# ─── PDF Download ──────────────────────────────────────────────────────────────


async def download_pdf_from_url(url: str) -> tuple[bytes, str]:
    """
    Download a PDF from a publicly readable URL (e.g. Firebase Storage).
    Returns (file_bytes, inferred_filename).
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    # Try to infer a filename from the URL or Content-Disposition header
    content_disp = response.headers.get("content-disposition", "")
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip('"').strip("'")
    else:
        # Extract from URL path, stripping query params
        url_path = url.split("?")[0]
        filename = url_path.split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            filename = "pitch_deck.pdf"

    return response.content, filename


# ─── Unstructured Partition ────────────────────────────────────────────────────


async def partition_pdf(
    pdf_bytes: bytes,
    filename: str,
    strategy: str = "vlm",
) -> list[dict[str, Any]]:
    """
    Send a PDF to Unstructured.io's Partition API and get back structured
    elements with type labels, text, and metadata (including page numbers).

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        filename:  Filename to pass to the API (helps with format detection).
        strategy:  Partition strategy. "vlm" (vision-language model) is best
                   for pitch decks; "hi_res" is an alternative with OCR + layout. 

    Returns:
        List of element dicts, each with keys like:
            - type: "Title" | "NarrativeText" | "Table" | "ListItem" | ...
            - text: the extracted text content
            - metadata: dict with page_number, coordinates, etc.
    """
    client = unstructured_client.UnstructuredClient(
        api_key_auth=UNSTRUCTURED_API_KEY,
    )

    # Choose the partitioning strategy
    if strategy == "vlm":
        strat = shared.Strategy.VLM
    elif strategy == "hi_res":
        strat = shared.Strategy.HI_RES
    else:
        strat = shared.Strategy.AUTO

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=pdf_bytes,
                file_name=filename,
            ),
            strategy=strat,
            # VLM-specific model settings (only used when strategy=VLM)
            vlm_model=VLM_MODEL if strategy == "vlm" else None,
            vlm_model_provider=VLM_MODEL_PROVIDER if strategy == "vlm" else None,
            # Language hint for OCR fallback
            languages=["eng"],
            # ── Page splitting ──
            # Treats every page/slide as its own unit so slide context is preserved
            split_pdf_page=True,
            # Continue even if a page fails (don't lose the whole deck)
            split_pdf_allow_failed=True,
            # Max parallelism for faster processing
            split_pdf_concurrency_level=SPLIT_PDF_CONCURRENCY,
        ),
    )

    # Use async partitioning for non-blocking I/O
    res = await client.general.partition_async(request=req)

    # res.elements is a list of element dicts
    elements = [element for element in res.elements]
    return elements


# ─── Slide Grouping ───────────────────────────────────────────────────────────


def group_elements_by_page(elements: list[dict[str, Any]]) -> dict[int, list[dict]]:
    """
    Group Unstructured elements by their page number (slide number).
    Returns a dict mapping page_number → list of elements on that page.
    """
    pages: dict[int, list[dict]] = defaultdict(list)
    for el in elements:
        # The page number lives in metadata.page_number
        metadata = el.get("metadata", {})
        page_num = metadata.get("page_number", 0)
        pages[page_num].append(el)

    return dict(sorted(pages.items()))


# ─── Element Filtering ─────────────────────────────────────────────────────────

# Element types to discard entirely — these waste LLM context with no useful text.
SKIP_ELEMENT_TYPES = {"Image", "FigureCaption", "PageBreak", "PageNumber"}


# ─── Markdown Rendering ───────────────────────────────────────────────────────


def element_to_markdown(el: dict[str, Any]) -> str:
    """
    Convert a single Unstructured element to a Markdown-formatted line
    with its semantic type as a label.

    Elements in SKIP_ELEMENT_TYPES are filtered out (returns "").

    Examples:
        ## Company Name              (for Title elements)
        > Some narrative text...     (for NarrativeText)
        | col1 | col2 |             (for Table, rendered as-is if HTML)
        - List item text             (for ListItem)
    """
    el_type = el.get("type", "Unknown")

    # Skip image/visual-only elements — they have no useful text for inference
    if el_type in SKIP_ELEMENT_TYPES:
        return ""

    text = (el.get("text") or "").strip()

    if not text:
        return ""

    if el_type == "Title":
        return f"## {text}"
    elif el_type == "Header":
        return f"### {text}"
    elif el_type == "NarrativeText":
        return text
    elif el_type == "ListItem":
        return f"- {text}"
    elif el_type == "Table":
        # Tables may come as HTML in metadata.text_as_html; fall back to plain
        metadata = el.get("metadata", {})
        html_table = metadata.get("text_as_html")
        if html_table:
            return f"**[Table]**\n{html_table}"
        return f"**[Table]** {text}"
    elif el_type == "Footer":
        return f"<sub>{text}</sub>"
    else:
        # UncategorizedText, Address, EmailAddress, Formula, etc.
        return f"[{el_type}] {text}"


def render_slide_markdown(
    slide_num: int,
    elements: list[dict[str, Any]],
) -> str:
    """
    Render all elements for a single slide into a Markdown section.
    """
    lines: list[str] = []
    lines.append(f"# Slide {slide_num}")
    lines.append("")

    for el in elements:
        md_line = element_to_markdown(el)
        if md_line:
            lines.append(md_line)

    lines.append("")  # Trailing blank line for readability
    return "\n".join(lines)


def render_full_markdown(
    pages: dict[int, list[dict[str, Any]]],
    filename: str = "pitch_deck.pdf",
) -> str:
    """
    Render the complete deck into a single Markdown document.
    """
    header = f"# Deck Distillation: {filename}\n\n"
    header += f"_Extracted {len(pages)} slides using Unstructured.io VLM strategy._\n\n"
    header += "---\n\n"

    slide_sections = []
    for page_num, elements in pages.items():
        slide_sections.append(render_slide_markdown(page_num, elements))

    return header + "\n---\n\n".join(slide_sections)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────


async def distill_deck(
    pdf_url: str,
    strategy: str = "vlm",
) -> str:
    """
    Full Deck Distillation pipeline (stateless, in-memory):
    1. Download the PDF from the given URL.
    2. Send it to Unstructured.io for AI-powered partitioning.
    3. Group elements by slide/page.
    4. Return structured Markdown.

    Args:
        pdf_url:    Publicly readable URL pointing to the pitch deck PDF.
        strategy:   "vlm" (recommended) or "hi_res" or "auto".

    Returns:
        The full Markdown string.
    """
    print(f"📥 Downloading PDF from URL...")
    pdf_bytes, filename = await download_pdf_from_url(pdf_url)
    print(f"   Downloaded: {filename} ({len(pdf_bytes):,} bytes)")

    print(f"🔬 Partitioning with Unstructured.io (strategy={strategy})...")
    elements = await partition_pdf(pdf_bytes, filename, strategy=strategy)
    print(f"   Extracted {len(elements)} elements")

    if not elements:
        raise ValueError("Unstructured.io returned 0 elements. The PDF may be empty, corrupted, or image-only.")

    print(f"📄 Grouping elements by slide...")
    pages = group_elements_by_page(elements)
    print(f"   Found {len(pages)} slides")

    print(f"📝 Rendering Markdown...")
    markdown = render_full_markdown(pages, filename=filename)

    print(f"✅ Deck distillation complete!")
    return markdown


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


async def _main():
    """CLI runner for quick testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deck Distillation: Extract structured Markdown from a pitch deck PDF"
    )
    parser.add_argument(
        "url",
        help="Publicly readable URL to the pitch deck PDF (e.g. Firebase Storage URL)",
    )
    parser.add_argument(
        "--strategy",
        choices=["vlm", "hi_res", "auto"],
        default="vlm",
        help="Unstructured partition strategy (default: vlm)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save the distilled .md file (default: ./output)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_output",
        help="Also print the Markdown to stdout",
    )
    args = parser.parse_args()

    markdown = await distill_deck(
        pdf_url=args.url,
        strategy=args.strategy,
        output_dir=args.output_dir,
    )

    if args.print_output:
        print("\n" + "=" * 60)
        print(markdown)


if __name__ == "__main__":
    asyncio.run(_main())
