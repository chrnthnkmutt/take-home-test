# Prioritized Fix List

## P1 - High Value Gaps

4. Add load testing artifacts.
   - Create a `locustfile.py` or `k6` script under `load_tests/`.
   - Document throughput, latency percentiles, bottlenecks, and scaling notes.

5. Add required architecture diagrams in the requested formats.
   - Provide `architecture/data_pipeline_diagram.png` or `.drawio`.
   - Provide `architecture/system_architecture_diagram.png` or `.drawio`.
   - Keep `architecture/ARCHITECTURE.md` as the written explanation.

6. Improve the monthly update story in the architecture docs.
   - Explain how incremental updates work.
   - Clarify how the system avoids full re-indexing when the catalog changes monthly.

## P2 - Bonus / Quality Improvements

7. Replace placeholder tests with real coverage.
   - Add focused tests for the API endpoints, tools, and agent routing logic.
   - Prioritize deterministic unit tests over broad integration tests.

8. Add a brief limitations / future work section if it is not already complete enough.
   - Call out what is not implemented.
   - Explain any mock behavior, fallback modes, or known constraints.

## Already Covered

- FastAPI API endpoints
- Three core tools: catalog RAG, web search, price analysis
- SQLite query history and feedback storage
- ChromaDB-based ingestion and vector search
- Architecture markdown documentation
- Example queries in the README
