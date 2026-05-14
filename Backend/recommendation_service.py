from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency fallback for partially configured envs.
    load_dotenv = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover - handled as unavailable at runtime.
    Groq = None

BACKEND_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BACKEND_DIR / ".env")

DEFAULT_RECOMMENDATION_MODEL = "llama-3.3-70b-versatile"
RECOMMENDATION_CATEGORIES = {
    "eye_contact",
    "posture",
    "stability",
    "self_touch",
    "expression",
    "emotion_presence",
    "overall",
}
RECOMMENDATION_PRIORITIES = {"low", "medium", "high"}
RESPONSE_PREVIEW_CHARS = 500

RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "areas_to_improve": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": sorted(RECOMMENDATION_CATEGORIES),
                    },
                    "priority": {
                        "type": "string",
                        "enum": sorted(RECOMMENDATION_PRIORITIES),
                    },
                    "message": {"type": "string"},
                },
                "required": ["category", "priority", "message"],
                "additionalProperties": False,
            },
        },
        "next_session_focus": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "overall_assessment",
        "strengths",
        "areas_to_improve",
        "next_session_focus",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an expert interview coaching assistant.
Use only the provided session summary data.
Be smart, specific, concise, and evidence-based.
Do not invent problems that are not supported by the scores, emotions, or event counts.
Do not criticize every metric.
If the final score is high and most metrics are strong, emphasize strengths and suggest at most one light refinement.
If one metric is meaningfully lower than the others, prioritize that metric.
If a metric is low and related events support it, make the recommendation more specific.
Use emotion data carefully: average emotion confidence is model confidence, not a user weakness.
Do not infer personality, mental health, honesty, stress level, psychological traits, or intent.
Do not merge surprise with fear.
Do not make unsupported claims.
Keep the tone professional, constructive, and useful for interview coaching.
Return only valid JSON matching the requested structure."""


def build_recommendation_input(
    *,
    session_id: str,
    session: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    averages = summary.get("averages") or {}
    emotion_summary = summary.get("emotion_summary") or {}
    return {
        "session_id": session_id,
        "metrics": {
            "final_session_score": averages.get("final"),
            "average_eye_contact": averages.get("eye_contact"),
            "average_posture": averages.get("posture"),
            "average_stability": averages.get("stability"),
            "average_self_touch": averages.get("self_touch"),
            "average_expression": averages.get("expression"),
            "analyzed_frame_count": summary.get("frame_count"),
            "paused_seconds": session.get("total_paused_seconds"),
        },
        "emotion_summary": {
            "dominant_emotion": emotion_summary.get("dominant_emotion"),
            "average_emotion_confidence": emotion_summary.get("average_confidence"),
            "positive_emotion_probability": emotion_summary.get("average_positive_prob"),
            "emotion_distribution": emotion_summary.get("distribution"),
        },
        "event_counts": summary.get("event_counts") or {},
    }


def _clean_string_list(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            cleaned.append(value.strip())
        if len(cleaned) >= limit:
            break
    return cleaned


def _validate_recommendations(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        print("[recommendations] validation failed", {"reason": "payload_not_object"})
        return None

    if "recommendations" in payload and isinstance(payload["recommendations"], dict):
        print("[recommendations] validation note", {"reason": "unwrapped_recommendations_object"})
        payload = payload["recommendations"]

    overall_assessment = payload.get("overall_assessment")
    if not isinstance(overall_assessment, str) or not overall_assessment.strip():
        print(
            "[recommendations] validation failed",
            {
                "reason": "missing_overall_assessment",
                "keys": sorted(str(key) for key in payload.keys()),
            },
        )
        return None

    strengths = _clean_string_list(payload.get("strengths"), limit=4)
    next_session_focus = _clean_string_list(payload.get("next_session_focus"), limit=3)
    if not isinstance(payload.get("strengths"), list):
        print("[recommendations] validation warning", {"reason": "strengths_not_list"})
    if not isinstance(payload.get("next_session_focus"), list):
        print("[recommendations] validation warning", {"reason": "next_session_focus_not_list"})

    areas_to_improve: list[dict[str, str]] = []
    raw_areas = payload.get("areas_to_improve")
    if isinstance(raw_areas, list):
        for item in raw_areas:
            if not isinstance(item, dict):
                print("[recommendations] validation warning", {"reason": "area_not_object"})
                continue
            category = item.get("category")
            priority = item.get("priority")
            message = item.get("message")
            if (
                isinstance(category, str)
                and category in RECOMMENDATION_CATEGORIES
                and isinstance(priority, str)
                and priority in RECOMMENDATION_PRIORITIES
                and isinstance(message, str)
                and message.strip()
            ):
                areas_to_improve.append(
                    {
                        "category": category,
                        "priority": priority,
                        "message": message.strip(),
                    },
                )
            else:
                print(
                    "[recommendations] validation warning",
                    {
                        "reason": "area_item_invalid",
                        "category": category,
                        "priority": priority,
                        "has_message": isinstance(message, str) and bool(message.strip()),
                    },
                )
            if len(areas_to_improve) >= 2:
                break
    elif raw_areas is not None:
        print("[recommendations] validation warning", {"reason": "areas_to_improve_not_list"})

    print(
        "[recommendations] validation succeeded",
        {
            "strength_count": len(strengths),
            "area_count": len(areas_to_improve),
            "focus_count": len(next_session_focus),
        },
    )
    return {
        "overall_assessment": overall_assessment.strip(),
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "next_session_focus": next_session_focus,
    }


def _extract_json_object(content: str | None) -> dict[str, Any] | None:
    if not content:
        print("[recommendations] parsing failed", {"reason": "empty_content"})
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print("[recommendations] parsing failed", {"reason": "json_decode_error", "detail": str(exc)})
        return None
    if not isinstance(parsed, dict):
        print("[recommendations] parsing failed", {"reason": "not_json_object"})
        return None
    print("[recommendations] parsing succeeded")
    return parsed


def generate_recommendations(recommendation_input: dict[str, Any]) -> dict[str, Any] | None:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_RECOMMENDATION_MODEL", DEFAULT_RECOMMENDATION_MODEL)
    session_id = recommendation_input.get("session_id")
    api_key_present = bool(api_key)
    sdk_available = Groq is not None

    print(
        "[recommendations] config",
        {
            "session_id": session_id,
            "groq_api_key_present": api_key_present,
            "groq_sdk_available": sdk_available,
            "model": model,
        },
    )

    if not api_key_present or not sdk_available:
        print(
            "[recommendations] fallback",
            {
                "session_id": session_id,
                "reason": "missing_groq_config",
                "groq_api_key_present": api_key_present,
                "groq_sdk_available": sdk_available,
            },
        )
        return None

    print("[recommendations] generation attempted", {"session_id": session_id, "model": model})
    try:
        client = Groq(api_key=api_key, timeout=12.0)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Create concise personalized interview coaching recommendations from this JSON data. "
                        "Return exactly the required JSON object.\n\n"
                        f"Schema: {json.dumps(RECOMMENDATION_SCHEMA, sort_keys=True)}\n\n"
                        f"Session data: {json.dumps(recommendation_input, sort_keys=True)}"
                    ),
                },
            ],
            temperature=0.2,
            max_completion_tokens=700,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        print("[recommendations] groq call succeeded", {"session_id": session_id})
        print(
            "[recommendations] raw response preview",
            {
                "session_id": session_id,
                "length": len(content or ""),
                "preview": (content or "")[:RESPONSE_PREVIEW_CHARS],
            },
        )
    except Exception as exc:
        print(
            "[recommendations] fallback",
            {
                "session_id": session_id,
                "reason": "groq_call_failed",
                "exception_type": type(exc).__name__,
                "detail": str(exc)[:240],
            },
        )
        return None

    parsed = _extract_json_object(content)
    recommendations = _validate_recommendations(parsed)
    if recommendations is None:
        print("[recommendations] fallback", {"session_id": session_id, "reason": "invalid_response"})
        return None

    print("[recommendations] succeeded", {"session_id": session_id})
    return recommendations
