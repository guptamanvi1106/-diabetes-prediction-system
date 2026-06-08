"""
app.py
------
Production-ready Flask API for the Diabetes Prediction System.

Endpoints
─────────
GET  /health              → service liveness check
POST /predict             → single JSON prediction
POST /predict-batch       → CSV file upload, bulk predictions
GET  /model-info          → metadata about the loaded model
"""

import io
import os
import time
import uuid
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, make_response

from utils import (
    get_logger,
    load_model,
    validate_single_input,
    predict_single,
    predict_dataframe,
    FEATURE_NAMES,
    UPLOAD_DIR,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app    = Flask(__name__)
logger = get_logger("diabetes_api")

# ── Manual CORS (no flask-cors dependency needed) ─────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = ALLOWED_ORIGINS
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = ALLOWED_ORIGINS
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp, 204


# ── Error helpers ─────────────────────────────────────────────────────────────
def error_response(message: str, status: int = 400, details=None) -> tuple:
    body = {"status": "error", "message": message}
    if details:
        body["details"] = details
    return jsonify(body), status


def success_response(data: dict, status: int = 200) -> tuple:
    data["status"] = "success"
    return jsonify(data), status


# ── Request logging middleware ────────────────────────────────────────────────
@app.before_request
def log_request():
    request.start_time = time.time()
    request.req_id = str(uuid.uuid4())[:8]
    logger.info(
        "[%s] %-6s %s  (from %s)",
        request.req_id, request.method, request.path,
        request.remote_addr,
    )

@app.after_request
def log_response(response):
    elapsed = (time.time() - getattr(request, "start_time", time.time())) * 1000
    logger.info(
        "[%s] → %d  %.1f ms",
        getattr(request, "req_id", "?"),
        response.status_code,
        elapsed,
    )
    return response


# ════════════════════════════════════════════════════════════════════════════
# Endpoint 1 – Health check
# ════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    """Liveness probe – also verifies the model files exist."""
    try:
        load_model()
        model_status = "loaded"
    except FileNotFoundError as exc:
        return error_response(str(exc), status=503)

    return success_response({
        "service": "Diabetes Prediction API",
        "version": "1.0.0",
        "model_status": model_status,
        "features": FEATURE_NAMES,
    })


# ════════════════════════════════════════════════════════════════════════════
# Endpoint 2 – Single prediction
# ════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "Pregnancies": 5, "Glucose": 166, ... }
    Returns: { "prediction": "Diabetic", "confidence": 0.85, ... }
    """
    # Parse JSON
    if not request.is_json:
        return error_response("Content-Type must be application/json")

    data = request.get_json(silent=True)
    if data is None:
        return error_response("Invalid JSON body")

    logger.debug("[%s] Input: %s", request.req_id, data)

    # Validate
    cleaned, errors = validate_single_input(data)
    if errors:
        return error_response(
            "Input validation failed",
            status=422,
            details=errors,
        )

    # Predict
    try:
        result = predict_single(cleaned)
        logger.info(
            "[%s] Prediction: %s  (confidence=%.2f)",
            request.req_id,
            result["prediction"],
            result["confidence"],
        )
        return success_response({
            "input":       cleaned,
            "prediction":  result["prediction"],
            "prediction_code": result["prediction_code"],
            "confidence":  result["confidence"],
            "probabilities": result["probabilities"],
        })

    except Exception as exc:
        logger.error("[%s] Prediction error: %s", request.req_id, traceback.format_exc())
        return error_response("Internal prediction error", status=500, details=str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Endpoint 3 – Batch prediction (CSV upload)
# ════════════════════════════════════════════════════════════════════════════
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    POST /predict-batch
    Accepts a multipart form upload with field name 'file' (CSV).

    Query params:
        ?download=true  → returns a downloadable CSV
        (default)       → returns JSON results

    The CSV must have column headers matching FEATURE_NAMES.
    """
    if "file" not in request.files:
        return error_response("No file part in request. Use field name 'file'.")

    file = request.files["file"]
    if file.filename == "":
        return error_response("No file selected.")

    if not file.filename.lower().endswith(".csv"):
        return error_response("Only CSV files are supported.")

    # Save upload
    filename  = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception as exc:
        return error_response(f"Could not parse CSV: {exc}")

    if df.empty:
        return error_response("Uploaded CSV is empty.")

    logger.info("[%s] Batch upload: %d rows, columns=%s", request.req_id, len(df), list(df.columns))

    # Check required columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return error_response(
            "CSV is missing required feature columns.",
            status=422,
            details={"missing_columns": missing, "required_columns": FEATURE_NAMES},
        )

    try:
        result_df = predict_dataframe(df)
    except Exception as exc:
        logger.error("[%s] Batch error: %s", request.req_id, traceback.format_exc())
        return error_response("Batch prediction failed", status=500, details=str(exc))

    logger.info(
        "[%s] Batch complete: %d predictions  (Diabetic=%d / Not Diabetic=%d)",
        request.req_id,
        len(result_df),
        (result_df["Prediction_Code"] == 1).sum(),
        (result_df["Prediction_Code"] == 0).sum(),
    )

    # ── Return as downloadable CSV ───────────────────────────────────────────
    if request.args.get("download", "false").lower() == "true":
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="diabetes_predictions.csv",
        )

    # ── Return as JSON ───────────────────────────────────────────────────────
    records = result_df.to_dict(orient="records")
    summary = {
        "total_records":     len(records),
        "diabetic_count":    int((result_df["Prediction_Code"] == 1).sum()),
        "not_diabetic_count":int((result_df["Prediction_Code"] == 0).sum()),
        "avg_confidence":    round(float(result_df["Confidence"].mean()), 4),
    }

    return success_response({
        "summary": summary,
        "results": records,
    })


# ════════════════════════════════════════════════════════════════════════════
# Endpoint 4 – Model info
# ════════════════════════════════════════════════════════════════════════════
@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns metadata about the trained model."""
    try:
        model, scaler = load_model()
        return success_response({
            "model_type":   type(model).__name__,
            "kernel":       getattr(model, "kernel", "N/A"),
            "n_classes":    int(model.n_support_.sum()) if hasattr(model, "n_support_") else "N/A",
            "support_vectors": {
                "not_diabetic": int(model.n_support_[0]) if hasattr(model, "n_support_") else "N/A",
                "diabetic":     int(model.n_support_[1]) if hasattr(model, "n_support_") else "N/A",
            },
            "features":     FEATURE_NAMES,
            "scaler_mean":  [round(float(m), 4) for m in scaler.mean_],
            "scaler_scale":  [round(float(s), 4) for s in scaler.scale_],
            "accuracy_on_test": "92.21%",    # recorded from training
        })
    except FileNotFoundError as exc:
        return error_response(str(exc), status=503)


# ════════════════════════════════════════════════════════════════════════════
# Endpoint 5 – Sample input helper
# ════════════════════════════════════════════════════════════════════════════
@app.route("/sample-input", methods=["GET"])
def sample_input():
    """Returns a sample JSON payload for testing /predict."""
    return success_response({
        "sample_payload": {
            "Pregnancies": 5,
            "Glucose": 166,
            "BloodPressure": 72,
            "SkinThickness": 19,
            "Insulin": 175,
            "BMI": 25.8,
            "DiabetesPedigreeFunction": 0.587,
            "Age": 51,
        },
        "usage": "POST this payload to /predict with Content-Type: application/json",
    })


# ── 404 / 405 handlers ───────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(_):
    return error_response("Endpoint not found", status=404)

@app.errorhandler(405)
def method_not_allowed(_):
    return error_response("Method not allowed", status=405)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Diabetes Prediction API on port %d …", port)
    load_model()   # eager load at startup
    app.run(host="0.0.0.0", port=port, debug=debug)
