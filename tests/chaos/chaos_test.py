"""Chaos testing runner for validating system resilience under failure conditions."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ChaosScenario:
    """Defines a single chaos test scenario."""

    name: str
    description: str
    expected_status: int | None = None
    timeout: float = 10.0


@dataclass
class ScenarioResult:
    """Result of a chaos test scenario."""

    name: str
    passed: bool
    duration_ms: float
    status_code: int | None = None
    error: str | None = None


@dataclass
class ChaosTestRunner:
    """Runs chaos scenarios against a running API to validate resilience."""

    base_url: str = "http://localhost:8000"
    api_key: str = "test-api-key"
    results: list[ScenarioResult] = field(default_factory=list)

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    async def test_service_outage(self, client: httpx.AsyncClient) -> ScenarioResult:
        """Simulate contacting a non-existent service endpoint."""
        scenario = ChaosScenario(name="service_outage", description="Request to non-existent endpoint returns error", expected_status=404)
        start = time.monotonic()
        try:
            resp = await client.get(f"{self.base_url}/api/v1/nonexistent", headers=self._headers(), timeout=scenario.timeout)
            duration = (time.monotonic() - start) * 1000
            passed = resp.status_code == scenario.expected_status
            return ScenarioResult(name=scenario.name, passed=passed, duration_ms=round(duration, 2), status_code=resp.status_code)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            return ScenarioResult(name=scenario.name, passed=False, duration_ms=round(duration, 2), error=str(exc))

    async def test_slow_response(self, client: httpx.AsyncClient) -> ScenarioResult:
        """Verify that the health endpoint responds within acceptable latency."""
        scenario = ChaosScenario(name="slow_response", description="Health endpoint must respond under 2s", expected_status=200, timeout=2.0)
        start = time.monotonic()
        try:
            resp = await client.get(f"{self.base_url}/api/v1/health", headers=self._headers(), timeout=scenario.timeout)
            duration = (time.monotonic() - start) * 1000
            passed = resp.status_code == scenario.expected_status and duration < 2000
            return ScenarioResult(name=scenario.name, passed=passed, duration_ms=round(duration, 2), status_code=resp.status_code)
        except httpx.TimeoutException:
            duration = (time.monotonic() - start) * 1000
            return ScenarioResult(name=scenario.name, passed=False, duration_ms=round(duration, 2), error="Request timed out (slow response)")
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            return ScenarioResult(name=scenario.name, passed=False, duration_ms=round(duration, 2), error=str(exc))

    async def test_partial_failure(self, client: httpx.AsyncClient) -> ScenarioResult:
        """Send a request with invalid payload and confirm graceful error handling."""
        scenario = ChaosScenario(name="partial_failure", description="Invalid payload returns 4xx not 5xx", expected_status=422)
        start = time.monotonic()
        try:
            resp = await client.post(
                f"{self.base_url}/api/v1/ask",
                headers={**self._headers(), "Content-Type": "application/json"},
                json={},
                timeout=scenario.timeout,
            )
            duration = (time.monotonic() - start) * 1000
            passed = 400 <= resp.status_code < 500
            return ScenarioResult(name=scenario.name, passed=passed, duration_ms=round(duration, 2), status_code=resp.status_code)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            return ScenarioResult(name=scenario.name, passed=False, duration_ms=round(duration, 2), error=str(exc))

    async def test_unauthorized_access(self, client: httpx.AsyncClient) -> ScenarioResult:
        """Verify that requests without API key are rejected."""
        scenario = ChaosScenario(name="unauthorized_access", description="Missing API key returns 401/403", expected_status=None)
        start = time.monotonic()
        try:
            resp = await client.get(f"{self.base_url}/api/v1/papers/", timeout=scenario.timeout)
            duration = (time.monotonic() - start) * 1000
            passed = resp.status_code in (401, 403)
            return ScenarioResult(name=scenario.name, passed=passed, duration_ms=round(duration, 2), status_code=resp.status_code)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            return ScenarioResult(name=scenario.name, passed=False, duration_ms=round(duration, 2), error=str(exc))

    async def run_all(self) -> list[ScenarioResult]:
        """Execute all chaos scenarios and return results."""
        self.results = []
        async with httpx.AsyncClient() as client:
            scenarios: list[Callable] = [
                self.test_service_outage,
                self.test_slow_response,
                self.test_partial_failure,
                self.test_unauthorized_access,
            ]
            for scenario_fn in scenarios:
                result = await scenario_fn(client)
                self.results.append(result)
                status = "PASS" if result.passed else "FAIL"
                logger.info(f"[{status}] {result.name} ({result.duration_ms:.1f}ms)")
        return self.results

    def print_report(self) -> None:
        """Print a formatted test report."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print("\n" + "=" * 60)
        print("CHAOS TEST REPORT")
        print("=" * 60)
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            line = f"  [{status}] {r.name} - {r.duration_ms:.1f}ms"
            if r.status_code is not None:
                line += f" (HTTP {r.status_code})"
            if r.error:
                line += f" - {r.error}"
            print(line)
        print("=" * 60)
        print(f"Results: {passed}/{total} passed")
        print("=" * 60)


async def main():
    """Entry point for running chaos tests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runner = ChaosTestRunner()
    await runner.run_all()
    runner.print_report()
    failed = sum(1 for r in runner.results if not r.passed)
    raise SystemExit(failed)


if __name__ == "__main__":
    asyncio.run(main())
