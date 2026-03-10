"""
Autofill Schema – Pydantic models mirroring StartupOnboardingData.

This file is the single source of truth for:
  1. The JSON Schema sent to the LLM via response_format (structured output)
  2. Validation of the LLM's response
  3. LLM guidance via Field descriptions ("Only provide if explicitly stated...")

The schema matches the frontend's StartupOnboardingData interface and its
constants (VERTICALS, STAGES, VALUE_DRIVERS, CUSTOMER_TYPES, PRICING_STRATEGIES,
REVENUE_METRICS) exactly – so the autofill response can be spread directly
into the onboarding form state.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums (must match constants.ts EXACTLY) ──────────────────────────────────

VERTICALS = [
    "AI/ML",
    "FinTech",
    "HealthTech",
    "SaaS",
    "E-commerce",
    "EdTech",
    "PropTech",
    "Climate/CleanTech",
    "Consumer",
    "Enterprise Software",
    "Cybersecurity",
    "Other",
    "Marketplace / Network",
    "Developer Tools",
    "BioTech / Life Sciences",
    "Web3 / Crypto",
]

STAGES = [
    "Idea/Concept",
    "Pre-seed",
    "Seed",
    "Seed+",
    "Series A",
    "Series A+",
    "Series B+",
    "Bootstrapped / Revenue-funded",
    "Series C+",
]

VALUE_DRIVER_VALUES = [
    "scalability",
    "severity",
    "unique-tech",
    "emotional",
    "adaptability",
    "other",
]

CUSTOMER_TYPES = ["B2B", "B2C"]

PRICING_STRATEGY_IDS = [
    "subscription",
    "transaction",
    "licensing",
    "advertising",
    "services",
    "other",
]

# Revenue metrics by category (all possible values the form can accept)
ALL_REVENUE_METRICS = [
    # SaaS
    "MRR", "ARR", "LTV", "CAC", "Churn Rate", "Net Revenue Retention",
    # Transaction
    "GMV", "Take Rate", "Transaction Volume", "Average Transaction Size",
    # Licensing
    "Revenue per License", "Renewal Rate", "Number of Licenses Sold",
    # Advertising
    "DAU/MAU", "CPM", "Ad Revenue per User", "Engagement Rate",
    # Services
    "Revenue per Project", "Utilization Rate", "Average Contract Value",
]


# ─── Sub-Models ────────────────────────────────────────────────────────────────


class Competitor(BaseModel):
    """A single competitor entry. Always return exactly 3 competitors."""
    name: Optional[str] = Field(
        default=None,
        description="Competitor company name. Only provide if explicitly mentioned in the source text.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this competitor does. Only provide if explicitly stated or clearly inferable.",
    )
    howYouDiffer: Optional[str] = Field(
        default=None,
        description="How the startup differentiates from this competitor. Only provide if explicitly stated.",
    )


# ─── Main Autofill Schema ─────────────────────────────────────────────────────


class StartupAutofillData(BaseModel):
    """
    Partial startup onboarding data extracted from a pitch deck.

    RULES FOR THE LLM:
    - Only include fields you can confidently extract from the source text.
    - Omit any field you are unsure about — do NOT guess or hallucinate.
    - Never return empty strings; omit the field entirely instead.
    - Match enum values EXACTLY as listed in each field's description.
    - Never return File-related fields (companyLogo, pitchdeck, etc.).
    """

    # ── Step 1: Company Info ──

    companyName: Optional[str] = Field(
        default=None,
        description="The startup's official company name. Only provide if clearly stated.",
    )
    website: Optional[str] = Field(
        default=None,
        description="The startup's website URL. Only provide if explicitly mentioned. Must be a valid URL or domain.",
    )
    linkedIn: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL for the startup. Only provide if explicitly mentioned do not substitute with founder's linkedin.",
    )
    vertical: Optional[str] = Field(
        default=None,
        description=(
            "The startup's industry vertical. MUST be exactly one of: "
            + ", ".join(f'"{v}"' for v in VERTICALS)
            + ". Pick the closest match based on the startup's description. Do not invent new values."
        ),
    )
    stage: Optional[str] = Field(
        default=None,
        description=(
            "The startup's current funding stage. MUST be exactly one of: "
            + ", ".join(f'"{s}"' for s in STAGES)
            + ". Infer from funding/raise information if not explicitly stated. Do not randomly guess."
        ),
    )
    location: Optional[str] = Field(
        default=None,
        description="City, State or City, Country where the startup is based. Only provide if mentioned.",
    )

    # ── Step 2: Overview ──

    companyOverview: Optional[str] = Field(
        default=None,
        description=(
            "A high-level overview of what the company does, its mission, and value proposition. "
            "MUST be at least 30 words. Synthesize from the pitch deck's intro, solution, and product slides. "
            "Write in third person (e.g. 'Fitly is...' not 'We are...'). Be comprehensive but concise."
        ),
    )

    # ── Step 3: Value Proposition ──

    currentPainPoint: Optional[str] = Field(
        default=None,
        description=(
            "The core problem the startup is solving. MUST be at least 20 words. "
            "Extract from the Problem slide. Write as a clear problem statement."
        ),
    )
    valueDrivers: Optional[list[str]] = Field(
        default=None,
        description=(
            "Array of value driver keys. Each MUST be exactly one of: "
            + ", ".join(f'"{v}"' for v in VALUE_DRIVER_VALUES)
            + ". Select based on which value propositions are emphasized in the deck."
        ),
    )
    valueDriverExplanations: Optional[str] = Field(
        default=None,
        description=(
            "JSON string mapping value driver key to a 1-2 sentence explanation. "
            "Format: '{\"unique-tech\": \"explanation...\", \"scalability\": \"explanation...\"}'. "
            "Keys MUST match the valueDrivers array. Only include explanations for drivers "
            "that are explicitly discussed in the deck."
        ),
    )

    # ── Step 4: Business Model ──

    customerType: Optional[list[str]] = Field(
        default=None,
        description=(
            "Array of customer types. Each MUST be exactly one of: "
            + ", ".join(f'"{c}"' for c in CUSTOMER_TYPES)
            + ". Infer from business model description (e.g. 'retailers' = B2B, 'users/shoppers' = B2C)."
        ),
    )
    customerTypeExplanation: Optional[str] = Field(
        default=None,
        description="Brief explanation of the customer segments. Only provide if clearly inferable.",
    )
    businessStructure: Optional[str] = Field(
        default=None,
        description="Business structure description (e.g. 'Platform connecting retailers and consumers'). Only if stated.",
    )

    # ── Step 5: Funding & Pricing ──

    previousInvestors: Optional[str] = Field(
        default=None,
        description="Names of previous investors. Only if explicitly listed in the deck.",
    )
    leadInvestor: Optional[str] = Field(
        default=None,
        description="Name of the lead investor. Only if explicitly stated.",
    )
    roundDetails: Optional[str] = Field(
        default=None,
        description="Details about the current funding round (e.g. 'Raising $500K pre-seed'). Only if stated.",
    )
    fundingUse: Optional[str] = Field(
        default=None,
        description="How the funding will be used. Summarize from the financials/ask slide if present.",
    )
    pricingStrategies: Optional[list[str]] = Field(
        default=None,
        description=(
            "Array of pricing strategy IDs. Each MUST be exactly one of: "
            + ", ".join(f'"{p}"' for p in PRICING_STRATEGY_IDS)
            + ". Map from the business model slide (e.g. 'pay per use' → \"transaction\", "
            "'Freemium/$9.99/month' → \"subscription\", 'Enterprise licensing' → \"licensing\")."
        ),
    )
    subscriptionType: Optional[str] = Field(
        default=None,
        description="Type/name of subscription offering (e.g. 'Freemium with premium tier'). Only if subscription pricing is mentioned.",
    )
    subscriptionBillingCycle: Optional[str] = Field(
        default=None,
        description="Billing cycle (e.g. 'Monthly', 'Annual'). Only if explicitly stated.",
    )
    subscriptionTiers: Optional[str] = Field(
        default=None,
        description="Description of subscription tiers. Only if explicitly described.",
    )
    transactionFeeType: Optional[str] = Field(
        default=None,
        description="Type of transaction fee (e.g. 'Commission per sale'). Only if transaction pricing is mentioned.",
    )
    transactionFeePercentage: Optional[str] = Field(
        default=None,
        description="Transaction fee percentage (e.g. '2-5%'). Only if explicitly stated.",
    )
    licensingModel: Optional[str] = Field(
        default=None,
        description="Licensing model details (e.g. '$50K-$200K/year based on size'). Only if licensing is mentioned.",
    )
    adRevenueModel: Optional[str] = Field(
        default=None,
        description="Advertising revenue model details. Only if advertising pricing is mentioned.",
    )
    serviceType: Optional[str] = Field(
        default=None,
        description="Service type details. Only if services pricing is mentioned.",
    )
    otherPricingDetail: Optional[str] = Field(
        default=None,
        description="Details for 'other' pricing strategy. Only if applicable.",
    )
    revenueMetrics: Optional[list[str]] = Field(
        default=None,
        description=(
            "Array of relevant revenue metrics. Each MUST be exactly one of: "
            + ", ".join(f'"{m}"' for m in ALL_REVENUE_METRICS)
            + ". Select metrics that match the startup's pricing strategies."
        ),
    )
    revenueMetricsValues: Optional[str] = Field(
        default=None,
        description="Actual values for revenue metrics if stated (e.g. 'MRR: $10K, CAC: $50'). ONLY include real numbers if found in the deck, otherwise don't fill",
    )

    # ── Step 5b: Go-to-Market ──

    gtmAcquisition: Optional[str] = Field(
        default=None,
        description="Go-to-market acquisition strategy. Summarize from GTM slide if present.",
    )
    gtmTimeline: Optional[str] = Field(
        default=None,
        description="Go-to-market timeline/phases. Summarize the key phases and timeframes if stated.",
    )

    # ── Step 6: Customer & Market ──

    targetGeography: Optional[str] = Field(
        default=None,
        description="Target geographic market (e.g. 'United States', 'Global'). Only if mentioned, if not infer from the deck and startup idea.",
    )
    targetCustomerDescription: Optional[str] = Field(
        default=None,
        description=(
            "Description of the target customer. MUST be at least 20 words if provided. "
            "Describe who the ideal customer is based on the deck."
        ),
    )
    tamValue: Optional[str] = Field(
        default=None,
        description="Total Addressable Market value (e.g. '$585M'). Use what is explicitly stated with a dollar figure in deck. If not stated, infer from the deck and startup idea.",
    )
    tamCalculationMethod: Optional[str] = Field(
        default=None,
        description="Don't fill this value",
    )
    tamBreakdown: Optional[str] = Field(
        default=None,
        description="Breakdown of TAM calculation. Only if detailed breakdown is provided. If tamValue was inferred, mention that here in this field.",
    )
    samValue: Optional[str] = Field(
        default=None,
        description="Serviceable Addressable Market value. Use what is explicitly stated with a dollar figure in deck. If not stated, infer from the deck and startup idea.",
    )
    samBreakdown: Optional[str] = Field(
        default=None,
        description="Breakdown of SAM. Only if provided. If samValue was inferred, mention that here in this field.",
    )
    somValue: Optional[str] = Field(
        default=None,
        description="Serviceable Obtainable Market value. Use what is explicitly stated with a dollar figure in deck. If not stated, infer from the deck and startup idea.",
    )
    somTimeframe: Optional[str] = Field(
        default=None,
        description="Timeframe for SOM target. If not stated explicitily don't fill this value.",
    )
    somBreakdown: Optional[str] = Field(
        default=None,
        description="Breakdown of SOM. Only if provided. If somValue was inferred, mention that here in this field.",
    )

    # ── Step 7: Competitors ──

    competitors: Optional[list[Competitor]] = Field(
        default=None,
        description=(
            "Exactly 3 competitor entries. If fewer than 3 competitors are found in the deck, "
            "pad the remaining entries with empty objects (all fields null). "
            "If no competitors section exists, omit this field entirely."
        ),
    )
    competitiveMoat: Optional[str] = Field(
        default=None,
        description="What makes the startup defensible against competitors. Only if explicitly discussed.",
    )

    # ── Confidence Metadata ──

    confidence: Optional[str] = Field(
        default=None,
        description=(
            "A JSON string of confidence scores (0.0 to 1.0) for each field you populated. "
            "Format: '{\"companyName\": 1.0, \"vertical\": 0.8, ...}'. "
            "1.0 = directly stated in the text, 0.7-0.9 = strongly implied, "
            "below 0.7 = do not include the field at all."
        ),
    )


# ─── AutofillResponse wrapper ─────────────────────────────────────────────────


class AutofillResponse(BaseModel):
    """The top-level response from the memo inferencer."""
    success: bool = Field(
        description="Whether the extraction was successful.",
    )
    data: StartupAutofillData = Field(
        description="The extracted startup onboarding fields. Only includes fields that were confidently extracted.",
    )


# ─── JSON Schema Export ────────────────────────────────────────────────────────


def get_json_schema() -> dict:
    """
    Export the AutofillResponse schema as a JSON Schema dict.
    This is what gets sent to the LLM via response_format.
    """
    return AutofillResponse.model_json_schema()


if __name__ == "__main__":
    """Quick test: print the schema to inspect it."""
    import json
    print(json.dumps(get_json_schema(), indent=2))
