"""
Memo Inferencer – Web Service.

A stateless FastAPI service that accepts a pitch deck PDF URL and returns
structured startup memo data (AutofillResponse) that can be spread directly
into a StartupOnboardingData form.

Endpoints:
    POST /infer  — Main endpoint. Takes {url: string}.
    GET  /health — Health check.

Error Codes:
    400 — Bad request (missing/invalid URL)
    422 — Unprocessable (PDF has no extractable text)
    502 — Upstream service failure (Unstructured.io or Gemini API)
    504 — Upstream service timeout
"""
from __future__ import annotations

import os
import time
import traceback

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deck_distiller import distill_deck
from memo_inferencer import infer_memo


# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Memo Inferencer",
    description="AI-powered startup memo extraction from pitch deck PDFs",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────────


class InferRequest(BaseModel):
    """Request body for the /infer endpoint."""
    url: str = Field(
        description="Publicly readable URL to the pitch deck PDF (e.g. Firebase Storage URL)",
    )


class InferResponse(BaseModel):
    """Response from the /infer endpoint."""
    success: bool
    data: dict
    timing: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "memo-inferencer"
    version: str = "1.0.0"


# ─── Error Helper ──────────────────────────────────────────────────────────────


def _raise(status_code: int, code: str, message: str):
    """Raise an HTTPException with a consistent error body."""
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Main inference endpoint (stateless — no file I/O).

    1. Downloads PDF from the given URL
    2. Sends it through Unstructured.io VLM for text extraction (in-memory)
    3. Passes the structured markdown to Gemini with strict schema
    4. Returns AutofillResponse JSON

    The response.data can be spread directly into Partial<StartupOnboardingData>.
    """
    # ── Validate input ──
    url = (request.url or "").strip()
    if not url:
        _raise(400, "MISSING_URL", "The 'url' field is required.")
    if not url.startswith("http"):
        _raise(400, "INVALID_URL", f"URL must start with http/https. Got: {url[:50]}")

    t_start = time.time()
    timing = {}

    # ── Step 1: Deck Distillation ──
    try:
        t1 = time.time()
        markdown = await distill_deck(pdf_url=url)
        timing["step1_distill_seconds"] = round(time.time() - t1, 2)
    except httpx.TimeoutException:
        _raise(504, "PDF_DOWNLOAD_TIMEOUT", "Timed out downloading the PDF. Is the URL accessible?")
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 404:
            _raise(400, "PDF_NOT_FOUND", f"PDF not found at the given URL (HTTP {status}).")
        elif status == 403:
            _raise(400, "PDF_ACCESS_DENIED", "Access denied. Is the PDF publicly readable?")
        else:
            _raise(502, "PDF_DOWNLOAD_FAILED", f"Failed to download PDF (HTTP {status}).")
    except ValueError as e:
        # Raised by distill_deck when extraction returns 0 elements
        _raise(422, "DISTILLATION_EMPTY", str(e))
    except Exception as e:
        traceback.print_exc()
        _raise(502, "DISTILLATION_FAILED", f"Unstructured.io processing failed: {type(e).__name__}: {e}")

    # ── Validate markdown has substance ──
    if not markdown or len(markdown.strip()) < 50:
        _raise(422, "DISTILLATION_EMPTY",
               "The PDF was processed but produced no usable text. It may be image-only or corrupted.")

    # ── Step 2: LLM Memo Inference ──
    try:
        t2 = time.time()
        result = await infer_memo(distilled_markdown=markdown)
        timing["step2_infer_seconds"] = round(time.time() - t2, 2)
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            _raise(502, "LLM_AUTH_FAILED", "LLM API key is invalid or missing.")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            _raise(504, "LLM_TIMEOUT", "LLM inference timed out. Try again.")
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            _raise(502, "LLM_RATE_LIMITED", "LLM API rate limit hit. Try again in a moment.")
        else:
            _raise(502, "LLM_FAILED", f"LLM inference failed: {type(e).__name__}: {e}")

    # ── Validate LLM result ──
    data = result.get("data", {})
    if not data or not data.get("companyName"):
        _raise(422, "INFERENCE_LOW_QUALITY",
               "The LLM could not extract meaningful data from this pitch deck. "
               "It may not contain enough text content.")

    timing["total_seconds"] = round(time.time() - t_start, 2)

    return InferResponse(
        success=True,
        data=data,
        timing=timing,
    )


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
