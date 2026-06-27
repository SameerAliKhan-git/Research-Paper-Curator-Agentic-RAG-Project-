# Load Testing

Load testing suite for the RAG API using [Locust](https://locust.io/).

## Quick Start

```bash
pip install locust
```

Run the load test:

```bash
locust -f tests/load/locustfile.py --host http://localhost:8000
```

Then open `http://localhost:8090` in your browser to access the Locust web UI.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LOCUST_API_KEY` | API key sent via `X-API-Key` header | (none) |

Example:

```bash
$env:LOCUST_API_KEY = "your-api-key"
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Web UI

1. Navigate to `http://localhost:8090` after starting Locust.
2. Enter the number of users to simulate and the spawn rate (users/second).
3. Click **Start** to begin the test.
4. Switch between the **Statistics**, **Charts**, **Failures**, and **Download Data** tabs to monitor progress.

## Headless Mode

Run without the web UI for CI pipelines:

```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 --run-time 3m
```

## Test Shape

The `FastLoadTestShape` class defines a staged load profile:

| Stage | Users | Spawn Rate | Duration | Purpose |
|---|---|---|---|---|
| 1 | 10 | 2/s | 0-60s | Warmup |
| 2 | 50 | 5/s | 60-180s | Sustained |
| 3 | 100 | 10/s | 180-300s | Peak |
| 4 | 20 | 2/s | 300-360s | Cooldown |

## Endpoints Tested

| Task | Endpoint | Weight |
|---|---|---|
| `ask_question` | `POST /api/v1/ask` | 5 |
| `hybrid_search` | `POST /api/v1/hybrid-search/` | 3 |
| `list_papers` | `GET /api/v1/papers/` | 2 |
| `paper_stats` | `GET /api/v1/papers/stats` | 1 |
| `health_check` | `GET /api/v1/health` | 1 |
