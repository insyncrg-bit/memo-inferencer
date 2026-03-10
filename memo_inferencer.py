"""
Memo Inferencer – Step 2: LLM-powered extraction from distilled pitch deck.

Takes the structured Markdown output from Deck Distillation (Step 1) and
passes it to an LLM with a strict system prompt + JSON schema to extract
startup memo fields matching the AutofillResponse structure.

Usage:
    from memo_inferencer import infer_memo
    result = await infer_memo(distilled_markdown)
    # result is a validated AutofillResponse dict
"""
from __future__ import annotations

from schema import (
    AutofillResponse,
    VERTICALS,
    STAGES,
    VALUE_DRIVER_VALUES,
    CUSTOMER_TYPES,
    PRICING_STRATEGY_IDS,
    ALL_REVENUE_METRICS,
)
from llm_client import create_llm_client


# ─── System Prompt ─────────────────────────────────────────────────────────────


def json_list(items: list[str]) -> str:
    """Format a list as a JSON array string for the prompt."""
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


SYSTEM_PROMPT = f"""You are an expert startup analyst AI. Your job is to extract structured data from a pitch deck that has been converted to Markdown.

You will receive a Markdown document organized slide-by-slide. Each slide has headings, narrative text, tables, and list items. Your task is to map this content to a structured JSON response matching the AutofillResponse schema.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — FOLLOW THESE EXACTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER HALLUCINATE. Only extract information that is explicitly stated or directly inferable from the text. If a field cannot be determined, set it to null — do NOT guess.

2. NEVER RETURN EMPTY STRINGS. If you don't have data for a field, use null, not "".

3. SUCCESS FIELD: Set "success" to true if you extracted at least companyName and one other field. Otherwise set it to false.

4. ENUM MATCHING — Use EXACT values from these lists:
   • vertical: {json_list(VERTICALS)}
   • stage: {json_list(STAGES)}
   • valueDrivers: {json_list(VALUE_DRIVER_VALUES)}
   • customerType: {json_list(CUSTOMER_TYPES)}
   • pricingStrategies: {json_list(PRICING_STRATEGY_IDS)}
   • revenueMetrics: {json_list(ALL_REVENUE_METRICS)}

5. WORD MINIMUMS — These fields have minimum word counts:
   • companyOverview: at least 30 words. Synthesize from the intro, solution, and product slides.
   • currentPainPoint: at least 20 words. Extract from the problem slide.
   • targetCustomerDescription: at least 20 words if provided.

6. COMPETITORS — Return exactly 3 competitor objects. If fewer are found, pad with {{"name": null, "description": null, "howYouDiffer": null}}.

7. PRICING STRATEGY MAPPING — Convert deck language to our enum IDs:
   • "Pay per use" / "Commission" / "Transaction fee" → "transaction"
   • "Subscription" / "Freemium" / "Monthly fee" → "subscription"
   • "Licensing" / "Enterprise licensing" → "licensing"
   • "Advertising" / "Ad-supported" → "advertising"
   • "Services" / "Consulting" → "services"

8. VALUE DRIVER MAPPING — Convert deck themes to our enum values:
   • Scalability, growth potential, efficiency → "scalability"
   • Urgency, pain severity, cost of problem → "severity"
   • Proprietary tech, AI, unique algorithms → "unique-tech"
   • Brand trust, social proof, peace of mind → "emotional"
   • Multi-market, cross-geography, versatile → "adaptability"

9. FUNDING STAGE INFERENCE — If the deck mentions raising amount:
   • Under $500K → "Pre-seed"
   • $500K - $2M → "Seed"
   • $2M - $5M → "Seed+"
   • $5M - $15M → "Series A"
   • $15M+ → "Series B+" or higher
   Only infer if the deck explicitly mentions a raise amount. If it says the stage name, use that directly.

10. CONFIDENCE SCORES — For each field you populate, add a confidence score in the "confidence" field:
    • 1.0 = directly quoted from the deck
    • 0.8-0.9 = strongly implied by context
    • 0.7 = reasonable inference
    • Below 0.7 = do NOT include the field

11. WRITING STYLE:
    • Write companyOverview in third person ("Fitly is..." not "We are...")
    • Keep descriptions factual and professional
    • Preserve specific numbers, percentages, and dollar amounts

12. FIELDS TO NEVER INCLUDE: companyLogo, logoPreview, startupLogoUrl, pitchdeck, pitchdeckName, pitchdeckUrl — these are handled by file uploads, not text extraction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION STRATEGY — Go slide by slide, ex-walkthrough:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Title/intro slide → companyName, website, tagline for companyOverview
• Problem slide → currentPainPoint
• Solution/Product slide → companyOverview (combine with intro), valueDrivers
• Market slide → tamValue, tamBreakdown, targetGeography, targetCustomerDescription
• Business Model slide → customerType, pricingStrategies, and their sub-fields
• GTM slide → gtmAcquisition, gtmTimeline
• Competitors slide → competitors array, competitiveMoat
• Financials/Ask slide → stage (infer), roundDetails, fundingUse
• Contact slide → website, linkedIn (if present)

Respond ONLY with a valid JSON object matching the AutofillResponse schema. No commentary, no markdown, no explanation."""


# ─── Inference Function ───────────────────────────────────────────────────────


async def infer_memo(
    distilled_markdown: str,
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.1,
) -> dict:
    """
    Step 2: Pass distilled deck markdown to an LLM and extract structured
    startup memo data.

    Args:
        distilled_markdown: The Markdown output from deck_distiller.distill_deck().
        provider:           LLM provider ("openai" or "gemini").
        model:              Model name (None = provider default).
        temperature:        LLM temperature (0.1 = deterministic extraction).

    Returns:
        Dict matching AutofillResponse schema:
        {
            "success": true,
            "data": { ...partial StartupOnboardingData... }
        }
    """
    client = create_llm_client(provider=provider, model=model)

    user_prompt = (
        "Below is a pitch deck that has been converted to structured Markdown. "
        "Extract as many startup memo fields as you can confidently identify.\n\n"
        "━━━ PITCH DECK CONTENT ━━━\n\n"
        f"{distilled_markdown}"
    )

    result = await client.structured_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=AutofillResponse,
        temperature=temperature,
        max_tokens=8192,
    )

    # Post-process: strip out null fields from data to keep it clean
    if "data" in result and isinstance(result["data"], dict):
        # Preserve confidence before stripping nulls
        confidence = result["data"].pop("confidence", None)

        result["data"] = {
            k: v for k, v in result["data"].items()
            if v is not None and v != "" and v != []
        }

        # Restore confidence if it was present
        if confidence is not None:
            result["data"]["confidence"] = confidence

        # Ensure competitors is exactly 3 entries if present
        if "competitors" in result["data"]:
            comps = result["data"]["competitors"]
            while len(comps) < 3:
                comps.append({"name": None, "description": None, "howYouDiffer": None})
            result["data"]["competitors"] = comps[:3]

    return result


# ─── Pretty Print Helper ──────────────────────────────────────────────────────


def format_autofill_result(result: dict) -> str:
    """Format the result for human-readable CLI output."""
    import json

    lines = []
    lines.append("=" * 60)
    lines.append("  AUTOFILL RESULT")
    lines.append("=" * 60)
    lines.append(f"\nSuccess: {result.get('success', False)}")

    data = result.get("data", {})
    confidence_raw = data.pop("confidence", None)
    # Parse confidence — may be a JSON string, a dict, or something unexpected
    confidence = {}
    if isinstance(confidence_raw, str):
        try:
            parsed = json.loads(confidence_raw)
            if isinstance(parsed, dict):
                confidence = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(confidence_raw, dict):
        confidence = confidence_raw

    lines.append(f"Fields extracted: {len(data)}\n")

    for key, value in data.items():
        conf = confidence.get(key, "?") if isinstance(confidence, dict) else "?"
        conf_str = f" (confidence: {conf})" if conf != "?" else ""

        if isinstance(value, list):
            if key == "competitors":
                lines.append(f"  {key}:{conf_str}")
                for i, comp in enumerate(value):
                    name = comp.get("name", "—")
                    lines.append(f"    [{i+1}] {name}")
            else:
                lines.append(f"  {key}: {value}{conf_str}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:{conf_str}")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        elif isinstance(value, str) and len(value) > 80:
            lines.append(f"  {key}:{conf_str}")
            lines.append(f"    {value[:120]}...")
        else:
            lines.append(f"  {key}: {value}{conf_str}")

    lines.append("")
    lines.append("─" * 60)
    lines.append("  Full JSON:")
    lines.append("─" * 60)
    # Restore confidence for JSON dump
    if confidence:
        data["confidence"] = confidence
    lines.append(json.dumps(result, indent=2, default=str))

    return "\n".join(lines)
