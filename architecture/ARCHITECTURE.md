# AI Product Research Assistant - Architecture Documentation

## System Overview

The AI Product Research Assistant is a sophisticated multi-agent system that intelligently routes natural language queries to specialized tools, aggregates results, and provides comprehensive answers. The system leverages Large Language Models (LLMs), vector databases, and web search to deliver accurate product research insights.

### Key Design Principles

1. **Intelligent Routing**: LLM-based query analysis determines which tools to use
2. **Modular Architecture**: Loosely coupled components for easy maintenance and scaling
3. **Graceful Degradation**: Fallback mechanisms ensure system reliability
4. **Idempotent Operations**: Safe to run operations multiple times
5. **Observability**: Comprehensive logging and metrics

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (HTTP Clients, curl, Postman, Web UI, Mobile Apps)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  /query  │  │ /queries │  │/feedback │  │ /health  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ProductResearchAgent                         │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  LLM-Based Router (Azure OpenAI)                  │  │  │
│  │  │  - Analyzes query intent                           │  │  │
│  │  │  - Selects appropriate tools                       │  │  │
│  │  │  - Determines execution order                      │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Tool Orchestrator                                 │  │  │
│  │  │  - Sequential tool execution                       │  │  │
│  │  │  - Result aggregation                              │  │  │
│  │  │  - Error handling                                  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────┬──────────────────┬──────────────────┬───────────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Tool 1:    │  │   Tool 2:    │  │   Tool 3:    │
│  Catalog RAG │  │  Web Search  │  │Price Analysis│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ChromaDB    │  │ Tavily API   │  │  CSV Data    │
│Vector Store  │  │(Web Search)  │  │  Analysis    │
└──────────────┘  └──────────────┘  └──────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│            Data Storage Layer                     │
│  ┌──────────────┐         ┌──────────────┐      │
│  │   SQLite     │         │  ChromaDB    │      │
│  │  (queries.db)│         │ (chroma_db/) │      │
│  │  - Queries   │         │  - Embeddings│      │
│  │  - Feedback  │         │  - Metadata  │      │
│  └──────────────┘         └──────────────┘      │
└──────────────────────────────────────────────────┘
```

## Component Architecture

### 1. API Layer (FastAPI)

**Location**: `src/api/main.py`, `src/api/models.py`

The API layer provides RESTful endpoints for client interaction.

#### Endpoints

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/query` | POST | Process query | `QueryRequest` | `QueryResponse` |
| `/queries` | GET | Get history | Query params | `QueryHistoryResponse` |
| `/feedback` | POST | Submit feedback | `FeedbackRequest` | `FeedbackResponse` |
| `/health` | GET | Health check | None | `HealthResponse` |
| `/` | GET | API info | None | JSON |

#### Key Features

- **CORS Support**: Configured for cross-origin requests
- **Automatic Documentation**: Swagger UI and ReDoc
- **Error Handling**: Global exception handler
- **Validation**: Pydantic models for request/response validation
- **Async Support**: Ready for async operations

#### Startup Process

1. Initialize SQLite database
2. Create ProductResearchAgent instance
3. Log system configuration
4. Start accepting requests

### 2. Agent Layer

**Location**: `src/agent/agent.py`

The agent is the brain of the system, responsible for intelligent query routing and result aggregation.

#### ProductResearchAgent

**Responsibilities:**
- Analyze query intent using LLM
- Select appropriate tools
- Execute tools in sequence
- Aggregate results
- Handle errors gracefully

**Routing Strategies:**

1. **LLM-Based Routing** (Preferred)
   - Uses Azure OpenAI
   - Analyzes query semantics
   - Provides reasoning for tool selection
   - Handles complex, multi-intent queries

2. **Rule-Based Routing** (Fallback)
   - Keyword matching
   - Fast and deterministic
   - Used when LLM unavailable

**Tool Selection Logic:**

```python
Query Type → Tools Used
─────────────────────────────────────────────────
"What products..." → ProductCatalogRAG
"Market price..." → WebSearchTool
"Profit margin..." → PriceAnalysisTool
"Should we adjust..." → All three tools (sequential)
```

**Execution Flow:**

```
1. Receive query
2. Analyze intent (LLM or rules)
3. Select tools
4. For each tool:
   a. Execute tool
   b. Collect results
   c. Handle errors
5. Aggregate results (LLM or concatenation)
6. Return final answer
```

### 3. Tools Layer

#### 3.1 Product Catalog RAG Tool

**Location**: `src/tools/product_catalog_rag.py`

**Purpose**: Semantic search over product catalog using RAG (Retrieval-Augmented Generation)

**Technology Stack:**
 - ChromaDB for vector storage
 - Azure OpenAI embeddings (or sentence-transformers fallback)
- LangChain for RAG pipeline

**Features:**
- Semantic similarity search
- Metadata filtering (category, price, rating, stock)
- Confidence scoring
- Context-aware answers

**Query Process:**
```
1. Embed query text
2. Search vector database
3. Apply metadata filters
4. Retrieve top-k results
5. Generate answer using LLM + context
6. Return products + answer
```

#### 3.2 Web Search Tool

**Location**: `src/tools/web_search.py`

**Purpose**: Search the web for market research and competitor information

**Technology Stack:**
- Tavily API for web search
- Mock search for development/testing

**Features:**
- Real-time web search
- Result ranking
- Content extraction
- Fallback to mock data

**Search Process:**
```
1. Format search query
2. Call Tavily API
3. Parse results
4. Extract relevant information
5. Rank by relevance
6. Return formatted results
```

#### 3.3 Price Analysis Tool

**Location**: `src/tools/price_analysis.py`

**Purpose**: Analyze product pricing, costs, and profitability

**Features:**
- Profit margin calculation
- Category-level analysis
- Low-margin product identification
- Pricing recommendations

**Analysis Process:**
```
1. Load product data from CSV
2. Calculate metrics:
   - Profit margin = (price - cost) / price * 100
   - Markup = (price - cost) / cost * 100
3. Filter by criteria
4. Aggregate by category
5. Generate insights
6. Return analysis + recommendations
```

### 4. Data Pipeline

**Location**: `src/pipeline/ingestion.py`, `src/pipeline/vector_store.py`

#### Data Flow

```
CSV File → Load → Validate → Transform → Embed → Store
   ↓         ↓        ↓          ↓         ↓       ↓
products  pandas   check    create    Azure OpenAI  ChromaDB
catalog   DataFrame schema   text     or ST   vector DB
```

#### Ingestion Process

1. **Load**: Read CSV with pandas
2. **Validate**: Check required columns
3. **Transform**: Create searchable text
4. **Embed**: Generate vector embeddings
5. **Store**: Save to ChromaDB with metadata

#### Monthly Catalog Updates

**Strategy**: Incremental updates with upsert operations

```python
# Full refresh (monthly)
ingest_products(csv_path="data/products_catalog.csv")

# Incremental update (as needed)
update_products(product_ids=["PROD-001", "PROD-002"])
```

**Update Process:**
1. Load new/updated products from CSV
2. Generate embeddings for changed products
3. Upsert to vector database (updates existing, adds new)
4. Old products remain unless explicitly deleted

**Benefits:**
- No downtime during updates
- Preserves existing data
- Efficient (only processes changes)
- Idempotent (safe to run multiple times)

### 5. Database Layer

#### SQLite Database

**Location**: `queries.db`, `products.db`

**Schema:**

```sql
-- Queries table
CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tools_used TEXT,  -- JSON array
    result TEXT,      -- JSON object
    response_time_ms REAL
);

-- Feedback table
CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    comment TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES queries(id)
);
```

**Operations:**
- `save_query()`: Store query and results
- `get_all_queries()`: Retrieve query history
- `save_feedback()`: Store user feedback

**Additional Local Artifacts:**
- Example scripts and helpers are provided under the `examples/` directory (e.g. `quick_start.py`, `run_agent.py`, `run_ingestion.py`, `test_api.sh`).
- Project tests are available in the `tests/` directory.

#### ChromaDB Vector Database

**Location**: `./chroma_db/`

**Collections:**
- `products`: Product embeddings and metadata

**Metadata Schema:**
```python
{
    "product_id": str,
    "name": str,
    "category": str,
    "brand": str,
    "price": float,
    "cost": float,
    "stock_status": str,  # in_stock, low_stock, out_of_stock
    "stock_quantity": int,
    "rating": float
}
```

## Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Programming language |
| FastAPI | Latest | Web framework |
| Uvicorn | Latest | ASGI server |
| LangChain | Latest | LLM framework |
| Azure OpenAI | Latest | LLM + Embeddings (hosted on Azure) |
| ChromaDB | Latest | Vector database |
| SQLAlchemy | Latest | Database ORM |
| Pandas | Latest | Data processing |
| Sentence Transformers | Latest | Embedding fallback |

### Development Tools

- **pytest**: Testing framework
- **python-dotenv**: Environment management
- **requests**: HTTP client
- **beautifulsoup4**: Web scraping (future use)

## Design Patterns

### 1. Strategy Pattern
- **Where**: Tool selection in agent
- **Why**: Different routing strategies (LLM vs rule-based)

### 2. Factory Pattern
- **Where**: Tool initialization
- **Why**: Create tools with different configurations

### 3. Singleton Pattern
- **Where**: Vector store, database connections
- **Why**: Reuse expensive resources

### 4. Chain of Responsibility
- **Where**: Tool execution pipeline
- **Why**: Sequential tool execution with error handling

### 5. Repository Pattern
- **Where**: Database operations
- **Why**: Abstract data access logic

## Scaling Strategy

### Horizontal Scaling

**Current State**: Single-instance deployment

**Scaling Path:**

1. **Phase 1: Stateless API** (Current)
   - API servers are stateless
   - Can run multiple instances behind load balancer
   - Shared database and vector store

2. **Phase 2: Distributed Vector Store**
   - Replace ChromaDB with Pinecone/Weaviate
   - Distributed vector search
   - Higher throughput

3. **Phase 3: Async Processing**
   - Queue-based architecture (Redis/RabbitMQ)
   - Background workers for long-running queries
   - WebSocket for real-time updates

4. **Phase 4: Microservices**
   - Separate services for each tool
   - Independent scaling
   - Service mesh (Istio)

### Caching Strategy

**Current**: No caching

**Proposed**:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Redis Cache │ ← Check cache first
└──────┬──────┘
       │ Cache miss
       ▼
┌─────────────┐
│  API Server │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Agent    │
└─────────────┘
```

**Cache Keys:**
- Query text (normalized)
- Tool results (TTL: 1 hour)
- Vector search results (TTL: 24 hours)

### Async Processing

**Current**: Synchronous execution

**Proposed**:

```python
# Parallel tool execution
async def process_query_async(query: str):
    tasks = []
    if needs_catalog:
        tasks.append(catalog_tool.query_async(query))
    if needs_web:
        tasks.append(web_tool.search_async(query))
    if needs_price:
        tasks.append(price_tool.analyze_async(query))
    
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

## Production Considerations

### Latency Optimization

**Current Performance:**
- Single tool query: 500-2000ms
- Multi-tool query: 1500-5000ms

**Optimization Strategies:**

1. **Caching**: 80% reduction for repeated queries
2. **Parallel Execution**: 50% reduction for multi-tool queries
3. **Embedding Cache**: 90% reduction for vector search
4. **Connection Pooling**: 20% reduction for database operations

**Target Performance:**
- Single tool: <500ms (p95)
- Multi-tool: <1500ms (p95)

### Cost Optimization

**Current Costs (per 1000 queries):**
- Azure OpenAI API: ~$0.10 (routing + aggregation)
- Azure OpenAI Embeddings: ~$0.02 (if used)
- Tavily Search: ~$1.00 (if used)

**Optimization Strategies:**

1. **Caching**: Reduce API calls by 70%
2. **Batch Processing**: Reduce embedding costs by 50%
3. **Smart Routing**: Use rule-based for simple queries
4. **Rate Limiting**: Prevent abuse

**Target Cost**: <$0.50 per 1000 queries

### Security

**Current Measures:**
- Environment variable for API keys
- Input validation with Pydantic
- SQL injection prevention (SQLAlchemy ORM)

**Production Requirements:**

1. **Authentication**: JWT tokens, API keys (store in Azure Key Vault)
2. **Authorization**: Role-based access control
3. **Rate Limiting**: Per-user/IP limits
4. **Encryption**: HTTPS, encrypted database
5. **Audit Logging**: Track all operations
6. **Input Sanitization**: Prevent injection attacks

### Monitoring

**Proposed Metrics:**

1. **Application Metrics**
   - Request rate (req/s)
   - Response time (p50, p95, p99)
   - Error rate (%)
   - Tool usage distribution

2. **Business Metrics**
   - Queries per user
   - Average feedback rating
   - Tool accuracy
   - Query success rate

3. **Infrastructure Metrics**
   - CPU/Memory usage
   - Database connections
   - Vector DB latency
   - API quota usage

**Monitoring Stack:**
- Prometheus for metrics
- Grafana for dashboards
- Sentry for error tracking
- ELK stack for log aggregation

## Design Trade-offs

### 1. Sequential vs Parallel Tool Execution

**Decision**: Sequential execution

**Rationale:**
- Simpler to implement and debug
- Some tools depend on others (e.g., catalog → web → price)
- Easier error handling
- Can add parallelism later

**Trade-off**: Higher latency for independent tools

### 2. LLM-Based vs Rule-Based Routing

**Decision**: LLM-based with rule-based fallback

**Rationale:**
- Better accuracy for complex queries
- Handles ambiguous intent
- Provides reasoning
- Graceful degradation

**Trade-off**: Higher cost and latency

### 3. ChromaDB vs Cloud Vector DB

**Decision**: ChromaDB (local)

**Rationale:**
- Easy setup for development
- No external dependencies
- Good performance for small-medium datasets
- Can migrate to cloud later

**Trade-off**: Limited scalability

### 4. SQLite vs PostgreSQL

**Decision**: SQLite

**Rationale:**
- Zero configuration
- Sufficient for development
- Easy to migrate
- File-based (portable)

**Trade-off**: No concurrent writes, limited scalability

### 5. Synchronous vs Asynchronous API

**Decision**: Synchronous with async-ready framework

**Rationale:**
- Simpler implementation
- FastAPI supports both
- Can add async later
- Tools are mostly I/O-bound

**Trade-off**: Lower throughput under high load

## Future Enhancements

### Short-term (1-3 months)

- [ ] Implement result caching (Redis)
- [ ] Add parallel tool execution
- [ ] Improve error messages
- [ ] Add request rate limiting
- [ ] Implement conversation history

### Medium-term (3-6 months)

- [ ] User authentication and authorization
- [ ] Migrate to cloud vector database (Pinecone)
- [ ] Add streaming responses
- [ ] Implement A/B testing framework
- [ ] Add multi-language support

### Long-term (6-12 months)

- [ ] Microservices architecture
- [ ] Custom tool registration system
- [ ] Machine learning for query intent
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration

## Deployment Architecture

### Development

```
Local Machine
├── Python virtual environment
├── SQLite database (queries.db)
├── ChromaDB (./chroma_db/)
└── Uvicorn server (localhost:8000)
```

### Production (Proposed)

```
┌─────────────────────────────────────────────┐
│              Load Balancer                   │
│            (AWS ALB / Nginx)                 │
└────────────┬────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐      ┌─────────┐
│ API     │      │ API     │
│ Server 1│      │ Server 2│
└────┬────┘      └────┬────┘
     │                │
     └────────┬───────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌─────────┐      ┌──────────────┐
│PostgreSQL│      │ Pinecone/    │
│ Database │      │ Weaviate     │
│ (RDS)    │      │ (Vector DB)  │
└──────────┘      └──────────────┘
```

### Kubernetes Deployment

```yaml
# Simplified example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-research-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: product-research-api
  template:
    metadata:
      labels:
        app: product-research-api
    spec:
      containers:
      - name: api
        image: product-research-api:latest
        ports:
        - containerPort: 8000
        env:
            - name: AZURE_OPENAI_API_KEY
               valueFrom:
                  secretKeyRef:
                     name: api-secrets
                     key: azure-openai-key
```

## Conclusion

The AI Product Research Assistant is designed with modularity, scalability, and reliability in mind. The current architecture supports development and small-scale production deployments, with a clear path to scale horizontally and handle enterprise workloads.

Key strengths:
- Intelligent query routing
- Modular tool architecture
- Graceful degradation
- Clear scaling path

Areas for improvement:
- Add caching layer
- Implement async processing
- Enhance monitoring
- Add authentication

---



For questions or suggestions, please open an issue or contact the maintainers.