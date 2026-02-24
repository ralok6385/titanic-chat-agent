"""
main.py — FastAPI backend for the Titanic Dataset Chat Agent.

Endpoints
---------
GET  /                 → Health check with dataset summary
POST /ask              → Accept a question, return answer + optional chart
GET  /chart/age-histogram → Generate & return an age histogram
GET  /chart/{filename} → Serve a previously saved chart image
GET  /stats            → Return quick dataset statistics
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import answer_question, age_histogram, df, CHARTS_DIR

# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────
app = FastAPI(
    title="🚢 Titanic Dataset Chat Agent",
    description=(
        "A chatbot API that analyses the Titanic dataset and "
        "returns text answers with beautiful visualisations."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (or any other frontend) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    chart: str | None = None


class StatsResponse(BaseModel):
    total_passengers: int
    survived: int
    perished: int
    survival_rate: float
    average_age: float
    average_fare: float
    male_count: int
    female_count: int
    classes: dict


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """Return a health-check JSON response with dataset info."""
    return {
        "status": "ok",
        "service": "Titanic Dataset Chat Agent",
        "version": "2.0.0",
        "dataset_rows": len(df),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/ask", response_model=AnswerResponse, tags=["Chat"])
async def ask_question_endpoint(payload: QuestionRequest):
    """
    Accept a natural-language question about the Titanic dataset
    and return an answer with an optional chart path.
    """
    try:
        result = answer_question(payload.question)
        return AnswerResponse(
            answer=result.get("answer", "No answer available."),
            chart=result.get("chart"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


@app.get("/stats", response_model=StatsResponse, tags=["Data"])
async def get_stats():
    """Return quick dataset statistics as JSON."""
    survived = int(df["survived"].sum())
    total = len(df)
    class_counts = df["pclass"].value_counts().sort_index().to_dict()
    return StatsResponse(
        total_passengers=total,
        survived=survived,
        perished=total - survived,
        survival_rate=round(survived / total * 100, 2),
        average_age=round(df["age"].mean(), 2),
        average_fare=round(df["fare"].mean(), 2),
        male_count=int(len(df[df["sex"] == "male"])),
        female_count=int(len(df[df["sex"] == "female"])),
        classes={f"class_{k}": v for k, v in class_counts.items()},
    )


@app.get("/chart/age-histogram", tags=["Charts"])
async def get_age_histogram():
    """Generate an age histogram and return the image file."""
    try:
        result = age_histogram.invoke("")
        chart_path = result.get("chart")
        if chart_path and os.path.isfile(chart_path):
            return FileResponse(chart_path, media_type="image/png")
        raise HTTPException(status_code=500, detail="Chart generation failed.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chart: {str(e)}",
        )


@app.get("/chart/{filename}", tags=["Charts"])
async def serve_chart(filename: str):
    """Serve a previously generated chart image by filename."""
    filepath = os.path.join(CHARTS_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(filepath, media_type="image/png")
