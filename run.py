"""
Memo Inferencer – CLI entry point (for local testing only).

The deployed service uses server.py instead.

Usage:
    python run.py <pitch_deck_url>
    python run.py <pitch_deck_url> --provider openai
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from deck_distiller import distill_deck
from memo_inferencer import infer_memo, format_autofill_result


async def run(pdf_url: str, provider: str = "gemini", model: str | None = None) -> dict:
    """Run the full pipeline: PDF URL → Markdown → AutofillResponse."""
    print("=" * 60)
    print("  MEMO INFERENCER (CLI)")
    print("=" * 60)
    print()

    # Step 1: Deck Distillation (in-memory)
    print("━━━ STEP 1: DECK DISTILLATION ━━━\n")
    markdown = await distill_deck(pdf_url=pdf_url)

    # Step 2: LLM Memo Inference
    print("\n━━━ STEP 2: LLM MEMO INFERENCE ━━━\n")
    result = await infer_memo(distilled_markdown=markdown, provider=provider, model=model)

    # Display
    print("\n" + format_autofill_result(result))

    # Save locally (CLI-only, not in deployed service)
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "autofill_result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n💾 Saved to: output/autofill_result.json")

    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Memo Inferencer CLI (local testing)")
    parser.add_argument("url", help="Pitch deck PDF URL")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None, help="Model override")
    args = parser.parse_args()

    asyncio.run(run(pdf_url=args.url, provider=args.provider, model=args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
