# AI Product Research Assistant

An intelligent product research system that uses AI agents and multiple specialized tools to answer natural language queries about products. The system intelligently routes queries to appropriate tools, aggregates results, and provides comprehensive answers.

## 🌟 Key Features

- **Intelligent Query Routing**: AI agent automatically selects the right tools based on query intent
- **Multi-Tool Integration**: 
  - Product Catalog RAG (Retrieval-Augmented Generation)
  - Web Search for market research
  - Price Analysis for profitability insights
- **Natural Language Interface**: Ask questions in plain English
- **RESTful API**: Easy integration with FastAPI
- **Vector Database**: Semantic search using ChromaDB
- **Query History & Feedback**: Track queries and collect user feedback

## 📋 Prerequisites

- **Python 3.8+** (tested with Python 3.9 and 3.10)
- **pip** package manager
- **API Keys** (at least one required):
  - Azure OpenAI credentials (for LLM + embeddings - recommended)
    - `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
  - OpenAI API Key (optional - for direct OpenAI usage or other integrations)
  - Tavily API Key (for web search - optional, has mock fallback)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd take-home-test

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` & `langchain-openai` - LLM integration (Azure/OpenAI)
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation (fallback)
- `pandas` - Data processing
- `sqlalchemy` - Database ORM
- `pytest` - Testing framework

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required configuration in `.env`:**

```bash
# Azure OpenAI configuration (highly recommended)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_azure_openai_deployment_name_here
# Optional: deployment name for embeddings (if different)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# OpenAI API Key (optional - for direct OpenAI usage or other integrations)
OPENAI_API_KEY=your_openai_api_key_here

# Tavily API Key (optional - for web search, has mock fallback)
TAVILY_API_KEY=your_tavily_api_key_here

# Database Configuration (default values work fine)
DATABASE_URL=sqlite:///./queries.db
```

**Getting API Keys:**
- **Azure OpenAI**: Get from Azure OpenAI service documentation (https://learn.microsoft.com/azure/cognitive-services/openai/)
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

### 4. Run Data Ingestion Pipeline

Load product data and generate embeddings:

```bash
python -m src.pipeline.ingestion
```

**What this does:**
1. Reads products from `data/products_catalog.csv`
2. Generates embeddings using Azure OpenAI (if configured) or sentence-transformers (fallback)
3. Stores embeddings in ChromaDB vector database (`./chroma_db/`)
4. Creates searchable product index

**Expected output:**
```
INFO - Loading products from data/products_catalog.csv
INFO - Loaded 50 products
INFO - Using Azure OpenAI embeddings (e.g. text-embedding-3-large)
INFO - Processing batch 1/1 (50 products)
INFO - Successfully ingested 50 products
INFO - Ingestion complete!
```

The ingestion command above is the same pipeline used by the Docker Compose setup.

### 5. Run the Application

You can start the project either locally or with Docker Compose.

**Docker Compose (recommended for a clean run):**

```bash
docker compose up --build
```

This starts the API container and runs ingestion in the companion service. After it is up, open:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

**Local development:**

```bash
uvicorn src.api.main:app --reload
```

If you have not already ingested the catalog locally, run:

```bash
python -m src.pipeline.ingestion
```

The API server starts at `http://localhost:8000`.

## 🧪 Load Testing

Load testing artifacts live under [load_tests](load_tests). Start with [load_tests/README.md](load_tests/README.md) for the Locust runner, the covered endpoints, and the metrics to capture.

## 📡 API Endpoints

### 1. POST /query - Process a Query

Submit a natural language query to the AI agent.

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What wireless headphones do we have under $200?"
  }' \
| jq
```

**Response:**
```json
{
  "query_id": "abc123",
  "query": "What wireless headphones do we have under $200?",
  "reasoning": "This query asks about products in our catalog...",
  "tools_used": ["ProductCatalogRAG"],
  "results": {
    "ProductCatalogRAG": {
      "answer": "Found 3 wireless headphones under $200...",
      "products": [...]
    }
  },
  "final_answer": "We have 3 wireless headphones under $200...",
  "metadata": {
    "timestamp": "2024-01-01T12:00:00.000Z",
    "execution_time_ms": 1234.56,
    "success": true
  }
}
```

### 2. GET /queries - Retrieve Query History

Get all previous queries with optional pagination.

**Request:**
```bash
# Get all queries
curl "http://localhost:8000/queries"

# With pagination
curl "http://localhost:8000/queries?limit=10&offset=0"
```

**Response:**
```json
{
  "queries": [
    {
      "id": "abc123",
      "query_text": "What wireless headphones...",
      "timestamp": "2024-01-01T12:00:00.000Z",
      "tools_used": ["ProductCatalogRAG"],
      "response_time_ms": 1234.56,
      "feedbacks": []
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

### 3. POST /feedback - Submit Feedback

Submit user feedback for a specific query.

**Request:**
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "abc123",
    "rating": 5,
    "comment": "Very helpful answer!"
  }' \
| jq
```

**Response:**
```json
{
  "feedback_id": "fb456",
  "query_id": "abc123",
  "rating": 5,
  "comment": "Very helpful answer!",
  "timestamp": "2024-01-01T12:05:00.000Z",
  "message": "Feedback submitted successfully"
}
```

### 4. GET /health - Health Check

Check system health and component status.

**Request:**
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "components": {
    "agent": {
      "status": "healthy",
      "message": "Agent running in llm mode"
    },
    "vector_database": {
      "status": "healthy",
      "message": "Vector database connected with 50 documents"
    },
    "llm": {
      "status": "healthy",
      "message": "LLM available for routing and aggregation"
    },
    "database": {
      "status": "healthy",
      "message": "Database connection successful"
    }
  },
  "version": "1.0.0"
}
```

## 💡 Example Queries

### 1. Product Catalog Search
ß
**Query:** "What wireless headphones do we have in stock?"

**Tools Used:** ProductCatalogRAG

**What it does:** Searches the product catalog using semantic search to find relevant products.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What wireless headphones do we have in stock?"}' \
| jq
```

### 2. Market Research

**Query:** "What's the current market price for Sony WH-1000XM5 headphones?"

**Tools Used:** WebSearchTool

**What it does:** Searches the web for current market prices and competitor information.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What'\''s the current market price for Sony WH-1000XM5 headphones?"}' \
| jq
```

### 3. Price Analysis

**Query:** "Which products have profit margins below 40%?"

**Tools Used:** PriceAnalysisTool

**What it does:** Analyzes product pricing, costs, and calculates profit margins.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which products have profit margins below 40%?"}' \
| jq
```

### 4. Comprehensive Analysis (Multi-Tool)

**Query:** "Should we lower the price of AudioMax Pro headphones based on competitor pricing?"

**Tools Used:** ProductCatalogRAG → WebSearchTool → PriceAnalysisTool

**What it does:** 
1. Gets product details from catalog
2. Researches competitor prices
3. Analyzes current margins
4. Provides pricing recommendation

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Should we lower the price of AudioMax Pro headphones based on competitor pricing?"}' \
| jq
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py
```

### Run Load Tests

```bash
pytest load_tests/ -v
```

### Test API with Script

```bash
# Make script executable
chmod +x examples/test_api.sh

# Run tests
./examples/test_api.sh
```

### Quick Start Example

```bash
# Run complete workflow example
python examples/quick_start.py
```

## 📁 Project Structure

```
take-home-test/
├── AGENT.md
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── architecture/
│   ├── Architecture.drawio
│   └── ARCHITECTURE.md
├── chroma_db/
│   ├── chroma.sqlite3
│   └── 879eb3ee-a2a4-41a4-952f-8b5ef255542e/
├── data/
│   └── products_catalog.csv
├── examples/
│   ├── quick_start.py
│   ├── run_agent.py
│   ├── run_ingestion.py
│   └── test_api.sh
├── load_tests/
│   ├── locustfile.py
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   └── vector_store.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── price_analysis.py
│   │   ├── product_catalog_rag.py
│   │   └── web_search.py
│   ├── database.py
│   ├── database_operations.py
│   └── models.py
├── test_chroma_db/
│   └── chroma.sqlite3
└── tests/
  ├── __init__.py
  ├── test_azure_migration.py
  ├── test_ingestion.py
  ├── test_price_analysis.py
  └── test_web_search.py
```

## File descriptions

- [AGENT.md](AGENT.md): Agent design, routing logic, and prompt-engineering notes.
- [Dockerfile](Dockerfile): Image build steps and runtime setup for the application.
- [docker-compose.yml](docker-compose.yml): Compose configuration to run the API and ingestion service.
- [README.md](README.md): Project overview, quick start, API docs, and usage instructions.
- [requirements.txt](requirements.txt): Python dependencies for running and testing the project.
- architecture/: Architecture diagrams and system design documentation (see [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)).
- chroma_db/: Persistent ChromaDB files used by the vector store (local DB snapshot and storage).
- data/products_catalog.csv: CSV of product records used for ingestion and semantic search.
- examples/quick_start.py: End-to-end example demonstrating ingestion → query → results.
- examples/run_agent.py: Minimal example showing how to call the agent programmatically.
- examples/run_ingestion.py: Script to run the ingestion pipeline locally.
- examples/test_api.sh: Simple curl-based script to exercise the API endpoints.
- load_tests/: Locust performance test scripts and README for load testing.
- src/__init__.py: Package marker for `src`.
- src/agent/agent.py: Core AI agent implementation that routes queries to tools and aggregates responses.
- src/api/main.py: FastAPI application and endpoint wiring (POST /query, GET /queries, etc.).
- src/api/models.py: Pydantic request/response models and validation schemas used by the API.
- src/pipeline/ingestion.py: Reads the products CSV, generates embeddings, and writes to ChromaDB.
- src/pipeline/vector_store.py: ChromaDB wrapper and utilities for creating/querying collections.
- src/tools/product_catalog_rag.py: Product-catalog retrieval tool (semantic search + context formatting).
- src/tools/web_search.py: Web-search tool (Tavily integration or mock fallback) for market data.
- src/tools/price_analysis.py: Pricing analysis utilities that compute margins and recommendations.
- src/database.py: SQLAlchemy engine and session setup (database connection configuration).
- src/database_operations.py: CRUD and helper functions for storing queries, feedback, and metadata.
- src/models.py: SQLAlchemy ORM models for queries, feedback, and products.
- test_chroma_db/chroma.sqlite3: Snapshot of a Chroma DB used in tests.
- tests/: Unit tests for ingestion, web search, price analysis, and Azure migration.

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | Recommended | - | Azure OpenAI key for LLM routing and embeddings |
| `AZURE_OPENAI_ENDPOINT` | Recommended | - | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Recommended | - | Azure OpenAI deployment name |
| `OPENAI_API_KEY` | Optional | - | OpenAI API key for other embedding providers |
| `TAVILY_API_KEY` | Optional | - | Tavily API key for web search |
| `DATABASE_URL` | No | `sqlite:///./queries.db` | Database connection string |
| `API_HOST` | No | `0.0.0.0` | API server host |
| `API_PORT` | No | `8000` | API server port |
| `DEBUG` | No | `False` | Enable debug mode |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Vector Database Configuration

The vector database is stored in `./chroma_db/` by default. You can customize this in the code:

```python
from src.pipeline.vector_store import get_vector_store

vector_store = get_vector_store(
    persist_directory="./custom_db",
    collection_name="products"
)
```

## ⚠️ Limitations

### Current Limitations

1. **Sequential Tool Execution**: Tools run one after another (no parallel execution)
2. **No Conversation History**: Each query is independent, no context from previous queries
3. **Limited to Catalog**: Product search limited to ingested catalog
4. **Web Search Dependency**: Web search quality depends on Tavily API or mock data
5. **Single Language**: Currently supports English only
6. **Rate Limiting**: No built-in rate limiting (should be added for production)
7. **Authentication**: No user authentication (should be added for production)

### Known Issues
- Using Gemini models may hit low rate limits; this repository currently uses OpenAI models hosted via Microsoft Foundry for both LLM and embedding tasks because they perform significantly better for our workloads. Note: using Foundry-hosted OpenAI incurs additional usage costs and requires committee approval/reimbursement.
- Generated responses may be longer than expected because no explicit token limit is configured for generation; system-prompt tuning and response-length controls are not yet implemented (optimization work pending).
- Large catalogs (>10,000 products) may require optimization
- Web search may be slow depending on network and API response times
- Vector database initialization takes time on first run

## 🚀 Future Improvements

### Short-term (Next Sprint)

- [ ] Add parallel tool execution for independent queries
- [ ] Implement result caching for frequently asked queries
- [ ] Add conversation history and context management
- [ ] Improve error messages and user feedback
- [ ] Add request rate limiting

### Medium-term (Next Quarter)

- [ ] Add user authentication and authorization
- [ ] Implement streaming responses for long-running queries
- [ ] Add support for multiple languages
- [ ] Create web UI for easier interaction
- [ ] Add product price tracking and alerts
- [ ] Implement A/B testing for different routing strategies

### Long-term (Future)

- [ ] Support for multiple product catalogs
- [ ] Integration with more e-commerce platforms
- [ ] Advanced analytics and reporting
- [ ] Machine learning for query intent prediction
- [ ] Custom tool registration and plugin system
- [ ] Distributed deployment with Kubernetes

## 📚 Additional Documentation

- [AGENT.md](AGENT.md) - Agent routing and tool selection notes
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) - System architecture and monthly update strategy

## 🎓 What We Learned

### Technical Insights

1. **LLM-based Routing**: Using Azure OpenAI for query routing provides much better accuracy than rule-based approaches, especially for complex queries
2. **Vector Databases**: ChromaDB provides excellent semantic search capabilities with minimal setup
3. **Tool Orchestration**: Sequential tool execution with result aggregation works well for most queries
4. **Fallback Strategies**: Having fallbacks (sentence-transformers for embeddings, mock for web search) ensures system reliability

### Challenges Faced

1. **Tool Selection Logic**: Determining when to use multiple tools vs. single tool required careful prompt engineering
2. **Result Aggregation**: Combining results from different tools in a coherent way needed LLM assistance
3. **Error Handling**: Ensuring graceful degradation when tools fail required comprehensive error handling
4. **Performance**: Balancing accuracy with response time, especially for multi-tool queries
5. **API Key Management**: Supporting multiple optional API keys while maintaining functionality

### Design Decisions

1. **Why Azure OpenAI**: Azure OpenAI deployments provide managed access to OpenAI models and simplify integration for routing decisions
2. **Why ChromaDB**: Easy to set up, good performance for small-medium datasets, persistent storage
3. **Why Sequential Execution**: Simpler to implement and debug; parallel execution can be added later
4. **Why SQLite**: Simple, no external dependencies, sufficient for development and small deployments
5. **Why FastAPI**: Modern, fast, automatic API documentation, async support

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## 🚢 Docker

Quick instructions to run the API and ingestion pipeline using Docker.

Build the image:

```bash
docker build -t ai-product-research:latest .
```

Start both the API and the ingestion job (the ingestion service runs once):

```bash
docker-compose up --build
```

After `docker-compose up` completes, the API will be available at http://localhost:8000

Notes:
- The compose file mounts the project directory for easy development. Remove mounts in `docker-compose.yml` for production images.
- ChromaDB persistence is stored under `./chroma_db` (mounted into the containers).


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the excellent LLM framework
- Microsoft/Azure for Azure OpenAI
- OpenAI for embeddings API (when using OpenAI directly)
- Tavily for web search API
- ChromaDB for vector database

---

For questions or issues, please open an issue on GitHub or contact the maintainers.