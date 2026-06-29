from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

from PIL import Image

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"

SCORE_SCALE_GUIDE = """
Use the full 1-10 scale consistently:
- 1-2: Critical issue; the element is missing, unreadable, misleading, or actively hurts performance.
- 3-4: Weak; the element is present but difficult to notice, understand, or trust.
- 5-6: Acceptable; the element works but needs clear improvement for paid media.
- 7-8: Good; the element is effective with only minor optimization opportunities.
- 9-10: Excellent; the element is immediately clear, polished, and highly suitable for conversion-focused media.
""".strip()

AI_SCORE_METRICS = [
    {
        "key": "product_visibility_score",
        "guide": "Score high if the product is large, clear, not blocked, and easy to identify.",
    },
    {
        "key": "brand_visibility_score",
        "guide": "Score high if the brand logo or brand name is clear and easy to notice.",
    },
    {
        "key": "text_readability_score",
        "guide": "Score high if important text can be read easily on mobile.",
    },
    {
        "key": "price_visibility_score",
        "guide": "Score high if the price is large, clear, and easy to find.",
    },
    {
        "key": "discount_visibility_score",
        "guide": "Score high if discount labels are easy to see and understand.",
    },
    {
        "key": "main_subject_focus_score",
        "guide": "Score high if the viewer immediately knows where to look first.",
    },
    {
        "key": "visual_hierarchy_score",
        "guide": "Score high if the viewer naturally sees brand, product, offer, and details in the right order.",
    },
    {
        "key": "layout_clarity_score",
        "guide": "Score high if the layout is clean, organized, and easy to scan.",
    },
    {
        "key": "background_distraction_score",
        "guide": "Score high if the background does not distract from the product.",
    },
    {
        "key": "creative_clutter_score",
        "guide": "Score high if the image is not too crowded.",
    },
    {
        "key": "message_clarity_score",
        "guide": "Score high if the main selling message is clear within 2 seconds.",
    },
    {
        "key": "cta_visibility_score",
        "guide": "Score high if there is a clear next action. If no call to action exists, give a low score.",
    },
    {
        "key": "premium_feel_score",
        "guide": "Score high if the creative looks professional, polished, and high quality.",
    },
    {
        "key": "trust_score",
        "guide": "Score high if the creative looks reliable, official, and not misleading.",
    },
    {
        "key": "attention_score",
        "guide": "Score high if the image is likely to stop someone scrolling.",
    },
    {
        "key": "platform_fit_score",
        "guide": "Score high if the image works well as a social media or ecommerce ad.",
    },
]


def get_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Run `pip install -r requirements.txt`."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    return OpenAI(api_key=api_key)


def prepare_image_for_ai(
    image_bytes: bytes,
    max_side: int = 1800,
) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        resampling_filter = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_side, max_side), resampling_filter)

        output = BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)

    return output.getvalue(), "image/jpeg"


def prepare_image_for_edit(
    image_bytes: bytes,
    max_side: int = 1800,
) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGBA")
        resampling_filter = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_side, max_side), resampling_filter)

        output = BytesIO()
        image.save(output, format="PNG", optimize=True)

    return output.getvalue()


def analyze_creative_with_openai(
    client: Any,
    image_bytes: bytes,
    model: str,
    target_format: str,
    platform_goal: str,
) -> dict[str, Any]:
    prepared_image, mime_type = prepare_image_for_ai(image_bytes)
    image_data_url = _to_data_url(prepared_image, mime_type)

    response = client.responses.create(
        model=model,
        instructions=_creative_scoring_system_prompt(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _creative_scoring_user_prompt(
                            target_format=target_format,
                            platform_goal=platform_goal,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        text=_json_schema_text_format(
            "electrolux_creative_scoring",
            _creative_scoring_response_schema(),
        ),
    )

    payload = _parse_json_response(_extract_response_text(response))
    return normalize_ai_scores(payload)


def consolidate_performance_with_openai(
    client: Any,
    model: str,
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    ai_scores: dict[str, Any],
    target_format: str,
    platform_goal: str,
) -> dict[str, Any]:
    input_payload = {
        "target_format": target_format,
        "platform_goal": platform_goal,
        "python_metrics": metrics,
        "python_metric_scores": basic_scores,
        "ai_creative_scores": ai_scores,
    }

    response = client.responses.create(
        model=model,
        instructions=_performance_system_prompt(),
        input=_performance_user_prompt(input_payload),
        text=_json_schema_text_format(
            "electrolux_performance_review",
            _performance_review_response_schema(),
        ),
    )

    payload = _parse_json_response(_extract_response_text(response))
    return normalize_performance_review(payload)


def generate_improved_image_with_openai(
    client: Any,
    image_bytes: bytes,
    image_model: str,
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    ai_scores: dict[str, Any],
    performance_review: dict[str, Any],
    target_format: str,
) -> dict[str, Any]:
    source_png = prepare_image_for_edit(image_bytes)
    source_file = BytesIO(source_png)
    source_file.name = "original_creative.png"

    prompt = _image_improvement_prompt(
        metrics=metrics,
        basic_scores=basic_scores,
        ai_scores=ai_scores,
        performance_review=performance_review,
        target_format=target_format,
    )

    response = client.images.edit(
        model=image_model,
        image=source_file,
        prompt=prompt,
        n=1,
        output_format="png",
        quality="high",
        input_fidelity="high",
        size="auto",
    )

    image_b64 = _extract_image_b64(response)
    return {
        "image_bytes": base64.b64decode(image_b64),
        "prompt": prompt,
        "model": image_model,
    }


def normalize_ai_scores(payload: dict[str, Any]) -> dict[str, Any]:
    raw_scores = payload.get("scores", {})

    if isinstance(raw_scores, list):
        raw_scores = {
            item.get("metric"): item
            for item in raw_scores
            if isinstance(item, dict) and item.get("metric")
        }

    if not isinstance(raw_scores, dict):
        raw_scores = {}

    normalized_scores = {}

    for metric in AI_SCORE_METRICS:
        key = metric["key"]
        raw_metric = raw_scores.get(key, {})

        if isinstance(raw_metric, dict):
            raw_score = raw_metric.get("score")
            raw_reason = raw_metric.get("reason", "")
        else:
            raw_score = raw_metric
            raw_reason = "No reason returned."

        normalized_scores[key] = {
            "score": _coerce_score(raw_score),
            "reason": str(raw_reason).strip() or "No reason returned.",
        }

    return {
        "scores": normalized_scores,
        "quick_read": str(payload.get("quick_read", payload.get("image_summary", ""))).strip(),
    }


def normalize_performance_review(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload.get("overall_performance", {})
    suggestions = payload.get("suggestions_to_improve", payload.get("suggestions", []))

    if not isinstance(overall, dict):
        overall = {"summary": str(overall)}

    if not isinstance(suggestions, list):
        suggestions = [suggestions]

    return {
        "overall_performance": {
            "overall_score": _coerce_score(overall.get("overall_score")),
            "performance_level": str(overall.get("performance_level", "Not rated")),
            "summary": str(overall.get("summary", "No summary returned.")),
            "strengths": _coerce_string_list(overall.get("strengths", [])),
            "risks": _coerce_string_list(overall.get("risks", [])),
        },
        "suggestions_to_improve": [_normalize_suggestion(item) for item in suggestions],
    }


def ai_score_average(ai_result: dict[str, Any]) -> float:
    scores = [item["score"] for item in ai_result.get("scores", {}).values()]

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 2)


def build_ai_score_rows(ai_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []

    for key, item in ai_result.get("scores", {}).items():
        rows.append(
            {
                "Metric": _format_metric_label(key),
                "Score": f"{item['score']}/10",
                "Reason": item["reason"],
            }
        )

    return rows


def _image_improvement_prompt(
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    ai_scores: dict[str, Any],
    performance_review: dict[str, Any],
    target_format: str,
) -> str:
    lowest_ai_scores = sorted(
        ai_scores.get("scores", {}).items(),
        key=lambda item: item[1].get("score", 10),
    )[:6]
    lowest_basic_scores = sorted(basic_scores.items(), key=lambda item: item[1])[:4]

    ai_focus = [
        {
            "metric": _format_metric_label(metric_key),
            "score": metric_value.get("score"),
            "reason": metric_value.get("reason"),
        }
        for metric_key, metric_value in lowest_ai_scores
    ]

    input_summary = {
        "target_format": target_format,
        "image_metrics": {
            "width": metrics.get("width"),
            "height": metrics.get("height"),
            "aspect_ratio": metrics.get("aspect_ratio"),
        },
        "lowest_python_scores": lowest_basic_scores,
        "lowest_ai_scores": ai_focus,
        "overall_performance": performance_review.get("overall_performance", {}),
        "suggestions_to_improve": performance_review.get("suggestions_to_improve", []),
    }

    return f"""
Create an improved version of the attached Electrolux media creative using the uploaded image as the source reference.

Goal: improve ecommerce performance while preserving the same product category, Electrolux brand feel, main offer intent, and visual credibility. Keep the output suitable for {target_format}.

Use this performance analysis as the design brief:
{json.dumps(input_summary, indent=2)}

Required improvements:
- Preserve recognizable product forms, brand colors, and professional appliance-retail style.
- Improve the lowest-scoring creative issues first, especially CTA visibility, text readability, clutter, focus, and hierarchy when they are low.
- Make the layout cleaner, easier to scan on mobile, and more conversion-oriented.
- Increase clarity of key offer elements without inventing unverifiable product claims.
- Add or strengthen a clear CTA if the analysis says CTA visibility is weak.
- Keep text areas readable and avoid tiny crowded copy.
- Reduce overexposed or distracting areas if technical metrics identify them.
- Do not create a dark, abstract, blurred, or purely decorative image.

Return only the improved image. Do not include explanations or extra text in the image unless they are part of the redesigned creative.
""".strip()


def _creative_scoring_system_prompt() -> str:
    metric_lines = "\n".join(
        f"- {metric['key']}: {metric['guide']}" for metric in AI_SCORE_METRICS
    )

    return f"""
You are a senior ecommerce and paid-social creative evaluator for Electrolux.
Evaluate only the uploaded image and only the visible evidence inside it. Do not assume brand assets, prices, offers, product claims, or call-to-action text that are not visible.
Your output is used directly by a Streamlit UI, so follow the response schema exactly and do not add extra keys, markdown, headings, code fences, or commentary.

{SCORE_SCALE_GUIDE}

For every metric, return:
- score: a numeric score from 1 to 10, using decimals only when useful.
- reason: one professional sentence explaining the visible evidence behind the score.

Scoring controls:
- If price is not visible, price_visibility_score must be 1-3.
- If discount or promotion information is not visible, discount_visibility_score must be 1-3.
- If no call to action is visible, cta_visibility_score must be 1-3.
- If text is present but too small or crowded for mobile scanning, text_readability_score must be 1-4.
- For background_distraction_score and creative_clutter_score, higher scores mean the image is cleaner, calmer, and less distracting.
- Do not give a score above 8 unless the image is clearly strong for that specific metric.

Metrics:
{metric_lines}
""".strip()


def _creative_scoring_user_prompt(target_format: str, platform_goal: str) -> str:
    keys = [metric["key"] for metric in AI_SCORE_METRICS]

    return f"""
Analyze this media creative for the target format "{target_format}" and placement goal "{platform_goal}".

Return JSON that matches the required schema exactly:
{{
    "quick_read": "One professional sentence summarizing the first impression and likely media performance.",
  "scores": {{
    "product_visibility_score": {{"score": 1, "reason": "..."}},
    "brand_visibility_score": {{"score": 1, "reason": "..."}},
    "text_readability_score": {{"score": 1, "reason": "..."}},
    "price_visibility_score": {{"score": 1, "reason": "..."}},
    "discount_visibility_score": {{"score": 1, "reason": "..."}},
    "main_subject_focus_score": {{"score": 1, "reason": "..."}},
    "visual_hierarchy_score": {{"score": 1, "reason": "..."}},
    "layout_clarity_score": {{"score": 1, "reason": "..."}},
    "background_distraction_score": {{"score": 1, "reason": "..."}},
    "creative_clutter_score": {{"score": 1, "reason": "..."}},
    "message_clarity_score": {{"score": 1, "reason": "..."}},
    "cta_visibility_score": {{"score": 1, "reason": "..."}},
    "premium_feel_score": {{"score": 1, "reason": "..."}},
    "trust_score": {{"score": 1, "reason": "..."}},
    "attention_score": {{"score": 1, "reason": "..."}},
    "platform_fit_score": {{"score": 1, "reason": "..."}}
  }}
}}

All required metric keys are: {keys}
Do not rename, remove, or add metric keys. Reasons must be specific enough for a marketer or designer to act on.
""".strip()


def _performance_system_prompt() -> str:
    return """
You are a senior performance creative strategist for Electrolux.
Create a professional AI conclusion from two inputs: Python technical image metrics and AI visual scoring.
Use the scores as evidence, especially the lowest-scoring metrics and any technical quality risks.
Be direct, commercially practical, and suitable for a business portal used by marketing and ecommerce teams.
Do not invent results, audience data, conversion rates, or platform performance claims that are not supported by the inputs.
Your output is rendered directly in the UI, so follow the response schema exactly and do not add extra keys, markdown, headings, code fences, or commentary.
""".strip()


def _performance_user_prompt(input_payload: dict[str, Any]) -> str:
    return f"""
Create the final media performance review from these inputs:

{json.dumps(input_payload, indent=2)}

Return JSON that matches the required schema exactly:
{{
  "overall_performance": {{
    "overall_score": 1,
        "performance_level": "Poor",
        "summary": "Two to three professional sentences explaining the total performance of this media.",
        "strengths": ["Specific strength based on the input metrics"],
        "risks": ["Specific weakness or performance risk based on the input metrics"]
  }},
  "suggestions_to_improve": [
    {{
      "priority": "High",
      "suggestion": "Specific design or content change.",
      "reason": "Why this should improve performance.",
      "based_on": ["metric_name_or_score_name"],
      "expected_impact": "What should improve after the change."
    }}
  ]
}}

The final response must have exactly two main sections: overall_performance and suggestions_to_improve.
Prioritize 3 to 6 suggestions that address the lowest scores first, using both AI creative scores and Python image scores.
Use only metric names present in the input for based_on.
Keep suggestions concrete, visual, and actionable, such as changing hierarchy, resizing text, improving product crop, clarifying offer, reducing clutter, improving contrast, or adding a CTA.
""".strip()


def _json_schema_text_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def _creative_scoring_response_schema() -> dict[str, Any]:
    score_properties = {
        metric["key"]: {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        }
        for metric in AI_SCORE_METRICS
    }

    return {
        "type": "object",
        "properties": {
            "quick_read": {"type": "string"},
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": [metric["key"] for metric in AI_SCORE_METRICS],
                "additionalProperties": False,
            },
        },
        "required": ["quick_read", "scores"],
        "additionalProperties": False,
    }


def _performance_review_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "overall_performance": {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "performance_level": {
                        "type": "string",
                        "enum": ["Poor", "Weak", "Okay", "Good", "Excellent"],
                    },
                    "summary": {"type": "string"},
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "overall_score",
                    "performance_level",
                    "summary",
                    "strengths",
                    "risks",
                ],
                "additionalProperties": False,
            },
            "suggestions_to_improve": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "suggestion": {"type": "string"},
                        "reason": {"type": "string"},
                        "based_on": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "expected_impact": {"type": "string"},
                    },
                    "required": [
                        "priority",
                        "suggestion",
                        "reason",
                        "based_on",
                        "expected_impact",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["overall_performance", "suggestions_to_improve"],
        "additionalProperties": False,
    }


def _format_metric_label(key: str) -> str:
    return key.removesuffix("_score").replace("_", " ").title()


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("OpenAI did not return valid JSON.")
        return json.loads(match.group(0))


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        output_items = response.get("output", [])
    else:
        output_items = getattr(response, "output", [])

    text_parts = []
    for output_item in output_items or []:
        content_items = (
            output_item.get("content", [])
            if isinstance(output_item, dict)
            else getattr(output_item, "content", [])
        )
        for content_item in content_items or []:
            text = (
                content_item.get("text")
                if isinstance(content_item, dict)
                else getattr(content_item, "text", None)
            )
            if text:
                text_parts.append(str(text))

    return "".join(text_parts)


def _extract_image_b64(response: Any) -> str:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)

    if not data:
        raise ValueError("OpenAI did not return an improved image.")

    first_image = data[0]
    image_b64 = (
        first_image.get("b64_json")
        if isinstance(first_image, dict)
        else getattr(first_image, "b64_json", None)
    )

    if not image_b64:
        raise ValueError("OpenAI image response did not include base64 image data.")

    return str(image_b64)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 1.0

    return round(max(1.0, min(10.0, score)), 2)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value]

    return []


def _normalize_suggestion(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {"suggestion": str(item)}

    return {
        "priority": str(item.get("priority", "Medium")),
        "suggestion": str(item.get("suggestion", "No suggestion returned.")),
        "reason": str(item.get("reason", "")),
        "based_on": _coerce_string_list(item.get("based_on", [])),
        "expected_impact": str(item.get("expected_impact", "")),
    }