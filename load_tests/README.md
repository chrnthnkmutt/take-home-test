# Load Testing

This directory contains the Locust load test used to exercise the FastAPI API under mixed traffic.

## What It Covers

- `GET /health` for low-cost readiness and baseline throughput.
- `GET /queries` for database read pressure.
- `POST /query` for the full agent path and tool orchestration.
- `POST /feedback` for the write path after a successful query.

## How to Run

```bash
pip install -r requirements.txt
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000
```

You can also point the test at a different host with `LOAD_TEST_HOST`:

```bash
LOAD_TEST_HOST=http://localhost:8000 locust -f load_tests/locustfile.py
```

## Interpreting the Results

Use the Locust stats panel and CSV export to record:

- Throughput: requests per second and sustained user concurrency.
- Latency percentiles: especially p50, p95, and p99 for each endpoint.
- Failure rate: request errors, timeouts, and non-200 responses.

## Likely Bottlenecks

- `/query` is the heaviest endpoint because it runs routing plus tool execution.
- Vector search and CSV-backed analysis can become noticeable once concurrent query traffic rises.
- Feedback writes are small, but they still depend on the SQLite write path.

## Scaling Notes

- Scale the API horizontally only after isolating the `/query` path, since it is much more expensive than `/health` and `/queries`.
- Keep ingestion out of the request path; it should remain a background or deployment-time task.
- If query latency rises first, look at model calls, vector search latency, and database contention before tuning the web tier.