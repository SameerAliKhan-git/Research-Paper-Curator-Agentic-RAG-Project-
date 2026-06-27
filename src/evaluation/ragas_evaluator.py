"""RAGAS evaluation for RAG pipeline quality metrics."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a RAGAS evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    answer_correctness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "context_relevancy": self.context_relevancy,
            "answer_correctness": self.answer_correctness,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        return (
            f"RAGAS Evaluation Results:\n"
            f"  Faithfulness:       {self.faithfulness:.3f}\n"
            f"  Answer Relevancy:   {self.answer_relevancy:.3f}\n"
            f"  Context Precision:  {self.context_precision:.3f}\n"
            f"  Context Recall:     {self.context_recall:.3f}\n"
            f"  Context Relevancy:  {self.context_relevancy:.3f}"
            + (f"\n  Answer Correctness: {self.answer_correctness:.3f}" if self.answer_correctness else "")
        )


class RAGASEvaluator:
    """RAGAS-based evaluation for RAG systems.

    Metrics:
    - Faithfulness: Does the answer stay true to the retrieved context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Are the retrieved chunks relevant and ranked well?
    - Context Recall: Does the context contain all needed information?
    - Context Relevancy: Is the retrieved context relevant to the question?
    - Answer Correctness: (Optional) Factual correctness vs ground truth
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        """Initialize RAGAS evaluator.

        Args:
            llm_model: LLM model for evaluation
            embedding_model: Embedding model for semantic similarity
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key

        self._evaluator = None
        self._initialized = False
        self._initialization_error = None

    def _initialize(self):
        """Lazy initialization of RAGAS evaluator."""
        if self._initialized:
            if self._initialization_error:
                raise self._initialization_error
            return

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            import os

            api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")

            if api_key and not api_key.startswith("your_openai_api_key") and not api_key.startswith("pk-"):
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                llm_kwargs = {"model": self.llm_model, "temperature": 0, "api_key": api_key}
                embed_kwargs = {"model": self.embedding_model, "api_key": api_key}
                self._llm = ChatOpenAI(**llm_kwargs)
                self._embeddings = OpenAIEmbeddings(**embed_kwargs)
                logger.info(f"RAGAS evaluator initialized with OpenAI model: {self.llm_model}")
            else:
                from langchain_ollama import ChatOllama, OllamaEmbeddings
                logger.warning("No valid OpenAI API key found for Ragas. Falling back to local Ollama.")
                eval_llm_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
                ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                
                # Fast connectivity and model validation check to avoid long timeouts
                import httpx
                try:
                    with httpx.Client(timeout=2.0) as client:
                        resp = client.get(f"{ollama_host}/api/tags")
                        if resp.status_code == 200:
                            models = [m["name"] for m in resp.json().get("models", [])]
                            llm_found = any(eval_llm_model in m or m.startswith(eval_llm_model.split(":")[0]) for m in models)
                            embed_found = any("nomic-embed-text" in m or m.startswith("nomic-embed-text") for m in models)
                            if not llm_found or not embed_found:
                                raise ValueError(
                                    f"Required Ollama models not found: LLM '{eval_llm_model}' (found: {llm_found}), "
                                    f"Embeddings 'nomic-embed-text' (found: {embed_found}). Available models: {models}"
                                )
                        else:
                            raise ValueError(f"Ollama returned status code {resp.status_code}")
                except Exception as check_err:
                    logger.error(f"Local Ollama model validation check failed: {check_err}")
                    raise RuntimeError(f"Ollama Ragas validation failed: {check_err}") from check_err

                self._llm = ChatOllama(model=eval_llm_model, temperature=0, base_url=ollama_host)
                self._embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)
                logger.info(f"RAGAS evaluator initialized with local Ollama: {eval_llm_model}")

            # Store metrics
            self._metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            self._answer_correctness_metric = answer_correctness
            self._evaluate_func = evaluate
            self._initialized = True

        except ImportError as e:
            self._initialized = True
            self._initialization_error = e
            logger.error(f"RAGAS or langchain dependencies not installed. Error: {e}")
            raise
        except Exception as e:
            self._initialized = True
            self._initialization_error = e
            logger.error(f"Failed to initialize RAGAS: {e}")
            raise

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single QA pair.

        Args:
            question: The question asked
            answer: The generated answer
            contexts: List of retrieved context chunks
            ground_truth: Optional ground truth answer for correctness metric

        Returns:
            EvaluationResult with all metrics
        """
        self._initialize()

        from datasets import Dataset
        # Prepare dataset for RAGAS
        dataset_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            dataset_dict["ground_truth"] = [ground_truth]
        dataset = Dataset.from_dict(dataset_dict)

        try:
            # Run evaluation
            result = self._evaluate_func(
                dataset=dataset,
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            # Extract scores
            scores = result.to_pandas().iloc[0]

            eval_result = EvaluationResult(
                faithfulness=float(scores["faithfulness"]),
                answer_relevancy=float(scores["answer_relevancy"]),
                context_precision=float(scores["context_precision"]),
                context_recall=float(scores["context_recall"]),
                context_relevancy=float(scores.get("context_relevancy", 0.0)),
                metadata={"question": question, "answer_length": len(answer), "num_contexts": len(contexts)},
            )

            # Add answer correctness if ground truth provided
            if ground_truth:
                try:
                    correctness_result = self._evaluate_func(
                        dataset=dataset,
                        metrics=[self._answer_correctness_metric],
                        llm=self._llm,
                        embeddings=self._embeddings,
                    )
                    correctness_scores = correctness_result.to_pandas().iloc[0]
                    eval_result.answer_correctness = float(correctness_scores["answer_correctness"])
                except Exception as e:
                    logger.warning(f"Answer correctness evaluation failed: {e}")

            return eval_result

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple QA pairs.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists for each question
            ground_truths: Optional list of ground truth answers

        Returns:
            List of EvaluationResult
        """
        self._initialize()

        from datasets import Dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if ground_truths:
            dataset_dict["ground_truth"] = ground_truths
        dataset = Dataset.from_dict(dataset_dict)

        try:
            results = self._evaluate_func(
                dataset=dataset,
                metrics=self._metrics + ([self._answer_correctness_metric] if ground_truths else []),
                llm=self._llm,
                embeddings=self._embeddings,
            )

            df = results.to_pandas()
            eval_results = []

            for i, row in df.iterrows():
                eval_result = EvaluationResult(
                    faithfulness=float(row["faithfulness"]),
                    answer_relevancy=float(row["answer_relevancy"]),
                    context_precision=float(row["context_precision"]),
                    context_recall=float(row["context_recall"]),
                    context_relevancy=float(row.get("context_relevancy", 0.0)),
                    answer_correctness=float(row["answer_correctness"])
                    if "answer_correctness" in row and ground_truths
                    else None,
                    metadata={
                        "question": questions[i],
                        "answer_length": len(answers[i]),
                        "num_contexts": len(contexts_list[i]),
                    },
                )
                eval_results.append(eval_result)

            return eval_results

        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            raise

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate statistics across multiple evaluations."""
        if not results:
            return {}

        return {
            "faithfulness_mean": sum(r.faithfulness for r in results) / len(results),
            "answer_relevancy_mean": sum(r.answer_relevancy for r in results) / len(results),
            "context_precision_mean": sum(r.context_precision for r in results) / len(results),
            "context_recall_mean": sum(r.context_recall for r in results) / len(results),
            "context_relevancy_mean": sum(r.context_relevancy for r in results) / len(results),
            "answer_correctness_mean": (
                sum(r.answer_correctness for r in results if r.answer_correctness is not None)
                / sum(1 for r in results if r.answer_correctness is not None)
                if any(r.answer_correctness is not None for r in results)
                else None
            ),
            "num_samples": len(results),
        }
