"""
src/vlm.py

Reusable Vision-Language Model (VLM) and Language Model utilities
for the Data Analytics with AI biomedical image-analysis assignment.

This module supports:
- Naive multimodal image prompting;
- Optimised structured VLM prompting with descriptive anchoring;
- Explicit uncertainty handling ("uncertain" token support);
- JSON extraction, validation, and schema enforcement;
- Repeated VLM runs for stochasticity analysis;
- Numbers-first LLM interpretation of quantitative image features;
- Reproducible local inference through Ollama with multi-model fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ollama


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

DEFAULT_VISION_MODEL = "llama3.2-vision"
FALLBACK_VISION_MODEL = "llava:7b"
SECONDARY_FALLBACK_VISION_MODEL = "moondream"
DEFAULT_TEXT_MODEL = "llama3.2"

NAIVE_TEMPERATURE = 0.7
STRUCTURED_TEMPERATURE = 0.0
NUMBERS_FIRST_TEMPERATURE = 0.0


# ---------------------------------------------------------------------
# Required JSON fields
# ---------------------------------------------------------------------

REQUIRED_VLM_FIELDS = [
    "modality",
    "tissue_type",
    "notable_features",
    "image_quality",
    "uncertainty",
    "summary_narrative",
]

REQUIRED_NUMBERS_FIRST_FIELDS = [
    "n_objects",
    "density_class",
    "shape_regularity",
    "quality_flag",
    "rationale",
]


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

NAIVE_PROMPT = """
Look at this microscopy image and describe what you see.
""".strip()

STRUCTURED_VLM_PROMPT = """
You are a biomedical imaging analysis assistant.

Your task is purely DESCRIPTIVE and OBJECTIVE.

Do NOT provide a clinical diagnosis, pathological staging,
disease classification, or speculative disease attribution.

Analyse only the information that is visually supported by the
supplied microscopy image.

Return a strictly valid JSON object containing exactly these fields:

{
    "modality": "fluorescence microscopy | light microscopy | other | uncertain",
    "tissue_type": "cellular nuclei | tissue section | culture | uncertain",
    "notable_features": [
        "signal intensity", "spatial distribution", "clustering"
    ],
    "image_quality": "high | moderate | degraded | uncertain",
    "uncertainty": "brief statement of any visual limitations or uncertain aspects",
    "summary_narrative": "concise 2-3 sentence factual description"
}

Rules:
1. Base every statement only on visually observable evidence.
2. Do not invent diagnoses or pathological findings.
3. Explicitly use "uncertain" whenever visual evidence is insufficient.
4. Return ONLY valid JSON, with no markdown fences or introductory chatter.
""".strip()

NUMBERS_FIRST_PROMPT = """
You are a biomedical data-analysis assistant auditing quantitative
measurements extracted from segmented microscopy images.

You will NOT see the original image.

You are given only the following numerical image-analysis results:

{feature_summary}

Use ONLY the supplied numerical measurements.

Do NOT invent visual findings, diagnoses, diseases, tissue identities,
or measurements that are not present in the data.

Generate:

1. A concise factual descriptive narrative interpreting the measured
   object count, density, morphology, and intensity information.

2. A structured JSON object with exactly these fields:

{{
    "n_objects": <integer>,
    "density_class": "<sparse | normal | dense | clustered | uncertain>",
    "shape_regularity": "<regular | moderate | irregular | uncertain>",
    "quality_flag": "<pass | warning | reject | uncertain>",
    "rationale": "<brief explanation based only on the supplied numbers>"
}}

Important rules:
- Base every statement solely on the supplied quantitative features.
- Do not claim that you directly observed the image.
- Do not provide clinical diagnosis or pathology.
- If the measurements are insufficient to support a conclusion, use "uncertain".
- The JSON values must be consistent with the narrative.

Output the descriptive narrative first, followed by the JSON object.
""".strip()


# ---------------------------------------------------------------------
# Input validation & JSON extraction
# ---------------------------------------------------------------------

def validate_image_path(image_path: Union[str, Path]) -> Path:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Expected an image file: {image_path}")
    return image_path


def _extract_content_from_response(response: Any) -> str:
    """Helper to extract text content across different ollama client versions."""
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return str(response.message.content).strip()
    elif isinstance(response, dict):
        msg = response.get("message", {})
        if isinstance(msg, dict):
            return str(msg.get("content", "")).strip()
        elif hasattr(msg, "content"):
            return str(msg.content).strip()
    return str(response).strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from raw text, markdown code fences, or truncated blocks."""
    if not text:
        return {"raw_output": text, "parse_error": True}

    # 1. Direct parse attempt
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. Clean markdown code fences and chatter
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    # 3. Find JSON section between outermost { and }
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # Try cleaning trailing commas
            fixed = re.sub(r",\s*(\}|\])", r"\1", candidate)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # 4. Handle truncated JSON where LLM omitted the final closing brace }
    if start != -1 and (end == -1 or end <= start):
        candidate = cleaned[start:]
        # Add closing brace
        for fix_candidate in [candidate + "\n}", candidate + "\"}"]:
            try:
                # Clean trailing comma if any
                clean_cand = re.sub(r",\s*$", "", fix_candidate.strip())
                parsed = json.loads(clean_cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # 5. Regex search for any valid JSON object block inside text
    json_match = re.search(r"(\{\s*\"[^{}]*\"\s*:\s*.*?\})", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {
        "raw_output": text,
        "parse_error": True
    }


def validate_vlm_json(response_json: Dict[str, Any]) -> Dict[str, Any]:
    if response_json.get("parse_error"):
        response_json = {}
    for field in REQUIRED_VLM_FIELDS:
        if field not in response_json:
            response_json[field] = "uncertain"
    return response_json


def validate_numbers_first_json(response_json: Dict[str, Any]) -> Dict[str, Any]:
    if response_json.get("parse_error"):
        response_json = {}
    for field in REQUIRED_NUMBERS_FIRST_FIELDS:
        if field not in response_json:
            response_json[field] = "uncertain"
    return response_json


# ---------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------

def query_vlm(
    image_path: Union[str, Path],
    prompt: str = STRUCTURED_VLM_PROMPT,
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = STRUCTURED_TEMPERATURE,
) -> Dict[str, Any]:
    """Send an image and prompt to local Ollama vision model."""
    image_path = validate_image_path(image_path)
    prompt_type = "structured" if prompt == STRUCTURED_VLM_PROMPT else "naive"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
            options={"temperature": temperature},
        )
        content = _extract_content_from_response(response)
        if not content:
            raise RuntimeError("The VLM returned an empty response.")

        structured_json = extract_json_from_text(content)
        return {
            "success": True,
            "raw_response": content,
            "json": structured_json,
            "requested_model": model,
            "actual_model": model,
            "model": model,
            "fallback_used": False,
            "fallback_reason": None,
            "prompt_type": prompt_type,
            "temperature": temperature,
        }
    except Exception as exc:
        err_msg = str(exc)
        # If llama3.2-vision encounters an error, reroute to llava:7b as specified
        if model == "llama3.2-vision" and model != FALLBACK_VISION_MODEL:
            print(f"[VLM Runtime Notice]: {model} encountered error ({err_msg[:60]}...). Rerouting to {FALLBACK_VISION_MODEL}...")
            try:
                fallback_res = query_vlm(
                    image_path=image_path,
                    prompt=prompt,
                    model=FALLBACK_VISION_MODEL,
                    temperature=temperature,
                )
                if fallback_res.get("success"):
                    fallback_res["requested_model"] = model
                    fallback_res["actual_model"] = fallback_res.get("actual_model", FALLBACK_VISION_MODEL)
                    fallback_res["model"] = fallback_res["actual_model"]
                    fallback_res["fallback_used"] = True
                    fallback_res["fallback_reason"] = f"Rerouted from {model} to {FALLBACK_VISION_MODEL} due to local engine error ({err_msg})"
                    return fallback_res
            except Exception as e2:
                print(f"[VLM Runtime Notice]: {FALLBACK_VISION_MODEL} also failed ({e2}). Rerouting to {SECONDARY_FALLBACK_VISION_MODEL}...")
            
            # Secondary fallback if llava is unavailable
            fallback_res = query_vlm(
                image_path=image_path,
                prompt=prompt,
                model=SECONDARY_FALLBACK_VISION_MODEL,
                temperature=temperature,
            )
            fallback_res["requested_model"] = model
            fallback_res["actual_model"] = fallback_res.get("actual_model", SECONDARY_FALLBACK_VISION_MODEL)
            fallback_res["model"] = fallback_res["actual_model"]
            fallback_res["fallback_used"] = True
            fallback_res["fallback_reason"] = f"Rerouted from {model} to {SECONDARY_FALLBACK_VISION_MODEL} due to local engine error ({err_msg})"
            return fallback_res

        return {
            "success": False,
            "error": err_msg,
            "raw_response": f"Ollama VLM Error: {exc}",
            "json": {"error": err_msg},
            "requested_model": model,
            "actual_model": model,
            "model": model,
            "fallback_used": False,
            "fallback_reason": None,
            "prompt_type": prompt_type,
            "temperature": temperature,
        }


def run_naive_vlm(
    image_path: Union[str, Path],
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = NAIVE_TEMPERATURE,
) -> Dict[str, Any]:
    return query_vlm(
        image_path=image_path,
        prompt=NAIVE_PROMPT,
        model=model,
        temperature=temperature,
    )


def run_structured_vlm(
    image_path: Union[str, Path],
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = STRUCTURED_TEMPERATURE,
) -> Dict[str, Any]:
    result = query_vlm(
        image_path=image_path,
        prompt=STRUCTURED_VLM_PROMPT,
        model=model,
        temperature=temperature,
    )
    if result["success"]:
        raw_json = result.get("json", {})
        is_valid_json = not raw_json.get("parse_error", False) and isinstance(raw_json, dict)
        is_complete = is_valid_json and all(k in raw_json for k in REQUIRED_VLM_FIELDS)
        
        result["json_valid"] = is_valid_json
        result["schema_complete"] = is_complete
        result["json"] = validate_vlm_json(raw_json)
    else:
        result["json_valid"] = False
        result["schema_complete"] = False
    return result


def evaluate_vlm_stochasticity(
    image_path: Union[str, Path],
    prompt: str = NAIVE_PROMPT,
    model: str = DEFAULT_VISION_MODEL,
    n_runs: int = 3,
    temperature: float = NAIVE_TEMPERATURE,
) -> List[Dict[str, Any]]:
    runs = []
    for i in range(1, n_runs + 1):
        res = query_vlm(
            image_path=image_path,
            prompt=prompt,
            model=model,
            temperature=temperature,
        )
        runs.append(
            {
                "run_index": i,
                "success": res["success"],
                "temperature": temperature,
                "model": model,
                "response": res["raw_response"],
                "json": res.get("json"),
            }
        )
    return runs


def query_text_model(
    prompt: str,
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Query a local Ollama LLM with text prompt."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = _extract_content_from_response(response)
        return {"success": True, "content": content, "model": model}
    except Exception as exc:
        return {"success": False, "content": f"Ollama Error: {exc}", "model": model, "error": str(exc)}


def query_llm_numbers_first(
    feature_summary: str,
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = NUMBERS_FIRST_TEMPERATURE,
) -> Dict[str, Any]:
    prompt = NUMBERS_FIRST_PROMPT.format(feature_summary=feature_summary)
    res = query_text_model(prompt=prompt, model=model, temperature=temperature)
    if not res["success"]:
        return {
            "success": False,
            "error": res.get("error"),
            "raw_response": res.get("content", ""),
            "json": {"error": res.get("error")},
            "model": model,
        }

    raw_text = res["content"]
    parsed_json = extract_json_from_text(raw_text)

    # If parsing failed or has parse_error, retry with strict zero-temperature JSON-only prompt
    if parsed_json.get("parse_error") or not all(k in parsed_json for k in REQUIRED_NUMBERS_FIRST_FIELDS):
        retry_prompt = f"""You are a biomedical data-analysis assistant.
Based ONLY on these numerical measurements:
{feature_summary}

Return ONLY one valid JSON object. Do not include any text, conversational chatter, or markdown fences before or after the JSON.
Follow this exact schema:
{{
    "n_objects": <integer count matching the data>,
    "density_class": "sparse | normal | dense | clustered | uncertain",
    "shape_regularity": "regular | moderate | irregular | uncertain",
    "quality_flag": "pass | warning | reject | uncertain",
    "rationale": "<brief 1-2 sentence explanation based only on the numbers>"
}}"""
        retry_res = query_text_model(prompt=retry_prompt, model=model, temperature=0.0)
        if retry_res.get("success"):
            retry_json = extract_json_from_text(retry_res["content"])
            if not retry_json.get("parse_error"):
                parsed_json = retry_json

    # Validate schema fields
    parsed_json = validate_numbers_first_json(parsed_json)

    return {
        "success": True,
        "raw_response": raw_text,
        "json": parsed_json,
        "model": model,
        "prompt_type": "numbers_first",
        "temperature": temperature,
    }


def get_vlm_configuration() -> Dict[str, Any]:
    return {
        "vision_model": DEFAULT_VISION_MODEL,
        "text_model": DEFAULT_TEXT_MODEL,
        "naive_temperature": NAIVE_TEMPERATURE,
        "structured_temperature": STRUCTURED_TEMPERATURE,
        "numbers_first_temperature": NUMBERS_FIRST_TEMPERATURE,
        "naive_prompt": NAIVE_PROMPT,
        "structured_vlm_prompt": STRUCTURED_VLM_PROMPT,
        "numbers_first_prompt": NUMBERS_FIRST_PROMPT,
        "required_vlm_fields": REQUIRED_VLM_FIELDS,
        "required_numbers_first_fields": REQUIRED_NUMBERS_FIRST_FIELDS,
    }