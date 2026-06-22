"""
DurianAI Backend API
FastAPI application for durian ripeness acoustic analysis + feedback collection.
"""
import os
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional

from services.audio_processor import analyze_audio, get_model_info, reload_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feedback storage directory
FEEDBACK_DIR = Path(__file__).parent / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DurianAI API",
    description="Durian ripeness analysis via acoustic (knock sound) classification + feedback collection",
    version="1.1.0",
)

# CORS — allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health Check
# ============================================================

@app.get("/api/health")
async def health():
    model_info = get_model_info()
    return {
        "status": "ok",
        "service": "DurianAI API",
        "version": "1.2.0",
        "feedback_count": _count_feedback_records(),
        "acoustic_model_loaded": model_info["loaded"],
        "acoustic_model_version": model_info.get("version"),
        "acoustic_inference_method": model_info.get("method", "heuristic"),
    }


@app.get("/api/model-status")
async def model_status():
    """
    Returns detailed model loading status for the acoustic model.
    Used by the frontend to show AI vs heuristic badge.
    """
    model_info = get_model_info()
    return {
        "acoustic_model_loaded": model_info["loaded"],
        "acoustic_model_version": model_info.get("version"),
        "acoustic_model_path": model_info.get("path"),
        "acoustic_inference_method": model_info.get("method", "heuristic"),
        "acoustic_labels": model_info.get("labels", ["unripe", "ripe", "overripe"]),
        "tflite_available": model_info.get("tflite_available", False),
        "feedback_count": _count_feedback_records(),
    }


@app.post("/api/model/reload")
async def reload_acoustic_model():
    """
    Hot-reload the acoustic model without restarting the server.
    Useful after deploying a new model file to backend/models/acoustic/.
    """
    try:
        success = reload_model()
        model_info = get_model_info()
        return {
            "status": "ok" if success else "fallback",
            "message": "模型重新載入成功" if success else "模型檔案不存在，使用啟發式",
            "acoustic_model_loaded": model_info["loaded"],
            "acoustic_inference_method": model_info.get("method", "heuristic"),
        }
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


def _count_feedback_records() -> int:
    """Count how many feedback records have been collected."""
    try:
        return len(list(FEEDBACK_DIR.glob("*.json")))
    except Exception:
        return -1


# ============================================================
# Acoustic Analysis
# ============================================================

@app.post("/api/analyze-acoustic")
async def analyze_acoustic(audio: UploadFile = File(...)):
    """
    Analyze a durian knock sound recording.
    
    Accepts: audio/webm, audio/ogg, audio/wav, audio/mp4 (any format librosa can decode)
    Returns: { ripeness, scores, confidence, method }
    """
    if not audio.content_type or not any(
        ct in audio.content_type
        for ct in ['audio/', 'application/octet-stream']
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {audio.content_type}. Expected audio file."
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Audio file too small (< 1KB). Please record a longer knock sound."
        )

    max_size = 10 * 1024 * 1024  # 10MB
    if len(audio_bytes) > max_size:
        raise HTTPException(status_code=413, detail="Audio file too large (> 10MB).")

    logger.info(f"Audio received: {audio.filename}, size={len(audio_bytes)}B, type={audio.content_type}")

    try:
        result = analyze_audio(audio_bytes)
        logger.info(f"Analysis: ripeness={result['ripeness']}, confidence={result['confidence']:.3f}")
        return JSONResponse({
            "ripeness": result["ripeness"],
            "scores": result["scores"],
            "confidence": result["confidence"],
            "method": result.get("method", "heuristic"),
        })
    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============================================================
# User Feedback Collection
# ============================================================

VALID_LABELS = {"unripe", "ripe", "overripe"}


class FeedbackPayload(BaseModel):
    ai_prediction: str
    user_label: str
    confidence: float
    acoustic_scores: Optional[dict] = None
    vision_scores: Optional[dict] = None
    timestamp: Optional[str] = None

    @field_validator("user_label", "ai_prediction")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Label must be one of {VALID_LABELS}, got '{v}'")
        return v


@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackPayload):
    """
    Receive user feedback after opening the durian.
    Stores: ai_prediction, user_label, scores, timestamp.
    
    This data is used for:
      1. Accuracy tracking (how often AI is correct)
      2. Future model fine-tuning with real-world labels
    """
    ts = payload.timestamp or datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": ts,
        "ai_prediction": payload.ai_prediction,
        "user_label": payload.user_label,
        "correct": payload.ai_prediction == payload.user_label,
        "confidence": payload.confidence,
        "acoustic_scores": payload.acoustic_scores,
        "vision_scores": payload.vision_scores,
    }

    # Save as individual JSON file (easy to process later)
    safe_ts = ts.replace(":", "-").replace(".", "-")[:23]
    filename = FEEDBACK_DIR / f"feedback_{safe_ts}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        logger.info(f"Feedback saved: {filename.name} | correct={record['correct']} | {payload.ai_prediction}→{payload.user_label}")
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        # Don't raise — frontend should succeed regardless of storage
        return JSONResponse({"status": "saved_failed", "message": str(e)}, status_code=200)

    # Compute running accuracy stats
    stats = _compute_feedback_stats()

    return JSONResponse({
        "status": "ok",
        "feedback_id": filename.stem,
        "correct": record["correct"],
        "stats": stats,
    })


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Return aggregated feedback statistics."""
    return JSONResponse(_compute_feedback_stats())


def _compute_feedback_stats() -> dict:
    """Compute accuracy stats from all saved feedback records."""
    records = []
    for f in FEEDBACK_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                records.append(json.load(fp))
        except Exception:
            continue

    if not records:
        return {"total": 0, "correct": 0, "accuracy": None, "by_class": {}}

    total = len(records)
    correct = sum(1 for r in records if r.get("correct", False))

    # Per-class accuracy
    by_class: dict[str, dict] = {}
    for r in records:
        cls = r.get("ai_prediction", "unknown")
        if cls not in by_class:
            by_class[cls] = {"total": 0, "correct": 0}
        by_class[cls]["total"] += 1
        if r.get("correct", False):
            by_class[cls]["correct"] += 1

    for cls in by_class:
        n = by_class[cls]["total"]
        c = by_class[cls]["correct"]
        by_class[cls]["accuracy"] = round(c / n, 3) if n > 0 else None

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total > 0 else None,
        "by_class": by_class,
    }


# ============================================================
# Feedback Export (for ML training)
# ============================================================

@app.get("/api/feedback/export")
async def export_feedback():
    """Export all feedback as a JSONL file for ML training."""
    records = []
    for f in sorted(FEEDBACK_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                records.append(json.load(fp))
        except Exception:
            continue

    # Return as JSONL
    jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    from fastapi.responses import Response
    return Response(
        content=jsonl,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=durian_feedback.jsonl"}
    )


# ============================================================
# Root / Docs
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "DurianAI API v1.1",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/analyze-acoustic",
            "/api/feedback (POST)",
            "/api/feedback/stats (GET)",
            "/api/feedback/export (GET)",
        ],
        "feedback_count": _count_feedback_records(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
