"""
Locust scenarios for the AI Product Research Assistant.

The mix below keeps a cheap readiness check in the foreground while still
including representative read and write traffic against the API.
"""

from os import getenv
from random import choice

from locust import HttpUser, between, task


class ProductResearchLoadTest(HttpUser):
    host = getenv("LOAD_TEST_HOST", "http://127.0.0.1:8000")
    wait_time = between(1, 3)

    def on_start(self):
        self.last_query_id = None

    @task(6)
    def health_check(self):
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")
                return

            payload = response.json()
            if payload.get("status") not in {"healthy", "degraded", "unhealthy"}:
                response.failure("Missing health status in response payload")

    @task(2)
    def query_history(self):
        with self.client.get(
            "/queries?limit=10&offset=0",
            name="GET /queries",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")
                return

            payload = response.json()
            if "queries" not in payload or "total" not in payload:
                response.failure("Unexpected query history payload")

    @task(3)
    def run_query(self):
        sample_queries = [
            "What wireless headphones do we have under $200?",
            "Show me products with strong profit margins in electronics.",
            "What do customers say about our best selling laptop accessories?",
        ]

        with self.client.post(
            "/query",
            json={"query": choice(sample_queries)},
            name="POST /query",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")
                return

            payload = response.json()
            if not payload.get("final_answer"):
                response.failure("Query response missing final_answer")
                return

            self.last_query_id = payload.get("query_id") or self.last_query_id

    @task(1)
    def submit_feedback(self):
        if not self.last_query_id:
            return

        with self.client.post(
            "/feedback",
            json={
                "query_id": self.last_query_id,
                "rating": 5,
                "comment": "Load test feedback sample",
            },
            name="POST /feedback",
            catch_response=True,
        ) as response:
            if response.status_code not in {200, 201}:
                response.failure(f"Expected 200 or 201, got {response.status_code}")
                return

            payload = response.json()
            if payload.get("message") != "Feedback submitted successfully":
                response.failure("Unexpected feedback response payload")