import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.dependencies import APIKeyDep, SessionDep
from src.evaluation.ragas_evaluator import RAGASEvaluator
from tests.evaluation.test_ragas import load_golden_dataset, evaluate_case, EvalCase, EvalResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

EVAL_LOG_PATH = "data/evaluation_results.json"

class EvaluationRunResponse(BaseModel):
    timestamp: str
    num_samples: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    was_fallback: bool
    details: List[Dict[str, Any]]

@router.post("/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    _key: APIKeyDep,
) -> EvaluationRunResponse:
    """Run RAGAS evaluation suite on golden dataset."""
    try:
        cases = load_golden_dataset()
        if not cases:
            raise HTTPException(status_code=400, detail="No evaluation cases found in golden dataset")

        results = []
        for case in cases:
            # Re-use pytest evaluation logic
            res = await evaluate_case(case)
            results.append(res)

        # Compute aggregates
        num_samples = len(results)
        faithfulness = sum(r.scores["faithfulness"] for r in results) / num_samples
        answer_relevancy = sum(r.scores["answer_relevancy"] for r in results) / num_samples
        context_precision = sum(r.scores["context_precision"] for r in results) / num_samples
        context_recall = sum(r.scores["context_recall"] for r in results) / num_samples
        was_fallback = any(r.was_fallback for r in results)

        details = []
        for r in results:
            details.append({
                "question": r.case.query,
                "answer": r.generated_answer,
                "contexts": r.retrieved_contexts,
                "scores": r.scores,
                "passed": bool(r.passed),
                "was_fallback": bool(r.was_fallback)
            })

        timestamp = datetime.now(timezone.utc).isoformat()
        run_data = {
            "timestamp": timestamp,
            "num_samples": num_samples,
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(context_recall, 4),
            "was_fallback": was_fallback,
            "details": details
        }

        # Save to file
        os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)
        history = []
        if os.path.exists(EVAL_LOG_PATH):
            try:
                with open(EVAL_LOG_PATH, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(run_data)
        with open(EVAL_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

        return EvaluationRunResponse(**run_data)

    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_evaluation_history(
    _key: APIKeyDep,
) -> List[Dict[str, Any]]:
    """Retrieve historical evaluation run statistics."""
    if not os.path.exists(EVAL_LOG_PATH):
        return []
    try:
        with open(EVAL_LOG_PATH, "r") as f:
            history = json.load(f)
        return history
    except Exception as e:
        logger.error(f"Failed to load evaluation history: {e}")
        return []
