"""Tests for the CostTracker service."""

import pytest

from src.services.cost_tracker import CostTracker, MODEL_PRICING


@pytest.fixture
def tracker():
    return CostTracker()


class TestTrackLlmCallRecordsCost:
    def test_track_llm_call_records_cost(self, tracker: CostTracker):
        record = tracker.track_llm_call(
            model="openai/gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert record.model == "openai/gpt-4o"
        assert record.prompt_tokens == 1000
        assert record.completion_tokens == 500
        assert record.cost_usd > 0
        assert record.timestamp is not None

    def test_cost_calculation_correct(self, tracker: CostTracker):
        # openai/gpt-4o: input_per_1k=0.0025, output_per_1k=0.01
        # Cost = (1000/1000)*0.0025 + (500/1000)*0.01 = 0.0025 + 0.005 = 0.0075
        record = tracker.track_llm_call(
            model="openai/gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert record.cost_usd == pytest.approx(0.0075, abs=1e-8)

    def test_metadata_stored(self, tracker: CostTracker):
        record = tracker.track_llm_call(
            model="llama3.2",
            prompt_tokens=100,
            completion_tokens=50,
            metadata={"source": "test"},
        )
        assert record.metadata == {"source": "test"}


class TestGetTotalCostAccumulates:
    def test_get_total_cost_accumulates(self, tracker: CostTracker):
        tracker.track_llm_call(
            model="openai/gpt-4o", prompt_tokens=1000, completion_tokens=500
        )
        tracker.track_llm_call(
            model="openai/gpt-4o-mini", prompt_tokens=2000, completion_tokens=1000
        )

        total = tracker.get_total_cost()
        # gpt-4o: 0.0025 + 0.005 = 0.0075
        # gpt-4o-mini: (2000/1000)*0.00015 + (1000/1000)*0.0006 = 0.0003 + 0.0006 = 0.0009
        expected = 0.0075 + 0.0009
        assert total == pytest.approx(expected, abs=1e-8)

    def test_total_zero_when_empty(self, tracker: CostTracker):
        assert tracker.get_total_cost() == 0.0


class TestGetCostByModelGroups:
    def test_get_cost_by_model_groups(self, tracker: CostTracker):
        tracker.track_llm_call(
            model="openai/gpt-4o", prompt_tokens=1000, completion_tokens=500
        )
        tracker.track_llm_call(
            model="openai/gpt-4o", prompt_tokens=1000, completion_tokens=500
        )
        tracker.track_llm_call(
            model="openai/gpt-4o-mini", prompt_tokens=2000, completion_tokens=1000
        )

        by_model = tracker.get_cost_by_model()
        assert "openai/gpt-4o" in by_model
        assert "openai/gpt-4o-mini" in by_model
        # Two gpt-4o calls: 2 * 0.0075 = 0.015
        assert by_model["openai/gpt-4o"] == pytest.approx(0.015, abs=1e-8)


class TestFreeModelsHaveZeroCost:
    def test_free_models_have_zero_cost(self, tracker: CostTracker):
        for model in ["llama3.2", "llama3.1", "mistral", "qwen2.5", "deepseek-r1"]:
            record = tracker.track_llm_call(
                model=model, prompt_tokens=10000, completion_tokens=5000
            )
            assert record.cost_usd == 0.0, f"{model} should be free"

    def test_all_free_models_in_pricing(self):
        for model in ["llama3.2", "llama3.1", "llama3", "llama2", "mistral", "mixtral",
                       "codellama", "gemma2", "phi3", "qwen2.5", "deepseek-r1"]:
            pricing = MODEL_PRICING[model]
            assert pricing["input_per_1k"] == 0.0
            assert pricing["output_per_1k"] == 0.0
