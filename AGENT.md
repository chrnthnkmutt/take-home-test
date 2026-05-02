# AI Agent with Routing Logic

## Overview

The `ProductResearchAgent` is an intelligent agent that coordinates three specialized tools to answer product research queries. It autonomously analyzes user queries, selects the appropriate tool(s), executes them in the correct order, and aggregates results into comprehensive answers.

## Architecture

### Components

1. **ProductResearchAgent** - Main orchestrator
2. **ProductCatalogRAG** - Internal product catalog search
3. **WebSearchTool** - External market research
4. **PriceAnalysisTool** - Pricing and margin analysis

### Routing Logic

The agent uses two routing strategies:

#### 1. LLM-Based Routing (Preferred)
- Uses Azure OpenAI to analyze query intent
- Understands complex queries and context
- Provides detailed reasoning for tool selection
- Handles multi-step queries intelligently

#### 2. Rule-Based Routing (Fallback)
- Keyword-based tool selection
- Fast and deterministic
- Used when LLM is unavailable

## Features

### Intelligent Tool Selection

The agent analyzes queries and selects tools based on intent:

| Query Type | Tools Used | Example |
|------------|------------|---------|
| Product catalog | ProductCatalogRAG | "What wireless headphones do we have?" |
| Market research | WebSearchTool | "What's the market price for Sony headphones?" |
| Pricing analysis | PriceAnalysisTool | "Which products have low margins?" |
| Comprehensive | All three tools | "Should we lower AudioMax headphones price?" |

### Sequential Tool Execution

For complex queries, the agent executes tools in sequence:

1. **ProductCatalogRAG** → Get internal product data
2. **WebSearchTool** → Research competitor prices
3. **PriceAnalysisTool** → Analyze margins and profitability

### Result Aggregation

The agent combines results from multiple tools:
- Uses LLM to synthesize information (when available)
- Provides actionable recommendations
- Highlights key insights from each source

## Usage

### Basic Usage

```python
from src.agent.agent import ProductResearchAgent

# Initialize agent
agent = ProductResearchAgent()

# Process a query
result = agent.process_query("What wireless headphones do we have?")

# Access results
print(result["reasoning"])      # Why these tools were selected
print(result["tools_used"])     # List of tools used
print(result["final_answer"])   # Aggregated answer
print(result["results"])        # Detailed results from each tool
```

### Advanced Usage

```python
# Initialize with custom tools
from src.tools.product_catalog_rag import ProductCatalogRAG
from src.tools.web_search import WebSearchTool
from src.tools.price_analysis import PriceAnalysisTool

catalog_tool = ProductCatalogRAG(persist_directory="./custom_db")
web_tool = WebSearchTool(tavily_api_key="your_key")
price_tool = PriceAnalysisTool()

agent = ProductResearchAgent(
    catalog_tool=catalog_tool,
    web_tool=web_tool,
    price_tool=price_tool,
    # For Azure OpenAI, provide `azure_api_key`, `azure_endpoint`, and `azure_deployment_name`
    azure_api_key=None,         # Azure OpenAI key (or use env var AZURE_OPENAI_API_KEY)
    azure_endpoint=None,        # Azure OpenAI endpoint (or use env var AZURE_OPENAI_ENDPOINT)
    azure_deployment_name=None, # Deployment name for the model on Azure
    model_name=None,            # Deprecated for Azure; prefer azure_deployment_name
    temperature=0.3
)

# Process query
result = agent.process_query("Should we adjust pricing for Electronics?")
```

### Response Format

```python
{
    "query": "original user query",
    "reasoning": "Explanation of tool selection",
    "tools_used": ["ProductCatalogRAG", "WebSearchTool"],
    "results": {
        "ProductCatalogRAG": {
            "answer": "...",
            "products": [...],
            "confidence": 0.85,
            ...
        },
        "WebSearchTool": {
            "answer": "...",
            "results": [...],
            ...
        }
    },
    "final_answer": "Aggregated answer combining all results",
    "metadata": {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "execution_time_ms": 1234.56,
        "success": true
    }
}
```

## Example Queries

### Single Tool Queries

#### Catalog Search
```python
result = agent.process_query("What wireless headphones do we have in stock?")
# Uses: ProductCatalogRAG
# Returns: List of wireless headphones with details
```

#### Market Research
```python
result = agent.process_query("What's the current market price for Sony WH-1000XM5?")
# Uses: WebSearchTool
# Returns: Market prices and competitor information
```

#### Price Analysis
```python
result = agent.process_query("Which products have profit margins below 40%?")
# Uses: PriceAnalysisTool
# Returns: Products with low margins and analysis
```

### Multi-Tool Queries

#### Comprehensive Pricing Decision
```python
result = agent.process_query(
    "Should we lower the price of AudioMax Pro headphones based on competitors?"
)
# Uses: ProductCatalogRAG → WebSearchTool → PriceAnalysisTool
# Returns: Comprehensive analysis with recommendation
```

**Execution Flow:**
1. **ProductCatalogRAG**: Get AudioMax Pro details (price, cost, specs)
2. **WebSearchTool**: Research competitor prices for similar headphones
3. **PriceAnalysisTool**: Analyze current margins and profitability
4. **Aggregation**: Combine insights and provide recommendation

## Error Handling

The agent handles errors gracefully:

```python
# Tool failure - continues with other tools
result = agent.process_query("Complex query")
if not result["metadata"]["success"]:
    print(f"Error: {result['metadata']['error']}")

# Individual tool errors are captured
for tool_name, tool_result in result["results"].items():
    if "error" in tool_result:
        print(f"{tool_name} failed: {tool_result['error']}")
```

## Configuration

### Environment Variables

```bash
# Required for LLM-based routing and aggregation (Azure OpenAI)
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name

# Optional - for web search (uses mock if not provided)
TAVILY_API_KEY=your_tavily_api_key
```

### Initialization Parameters

```python
ProductResearchAgent(
    catalog_tool=None,           # Optional pre-initialized tool
    web_tool=None,               # Optional pre-initialized tool
    price_tool=None,             # Optional pre-initialized tool
    azure_api_key=None,          # Azure OpenAI key (or use env var AZURE_OPENAI_API_KEY)
    azure_endpoint=None,         # Azure OpenAI endpoint (or use env var AZURE_OPENAI_ENDPOINT)
    azure_deployment_name=None,  # Deployment name for the model on Azure
    temperature=0.3              # Lower = more deterministic routing
)
```

## Performance

### Execution Times

- **Single tool**: 500-2000ms (depends on tool and query complexity)
- **Multiple tools**: 1500-5000ms (sequential execution)
- **LLM routing**: +200-500ms overhead
- **Rule-based routing**: +10-50ms overhead

### Optimization Tips

1. **Use rule-based routing** for simple, predictable queries
2. **Pre-initialize tools** to avoid repeated initialization
3. **Adjust temperature** lower (0.1-0.3) for more consistent routing
4. **Cache results** for repeated queries (implement externally)

## Logging

The agent provides detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Agent logs include:
# - Query analysis and tool selection
# - Tool execution status
# - Execution times
# - Errors and warnings
```

## Testing

Run the example script:

```bash
python examples/run_agent.py
```

This demonstrates:
- Single tool queries
- Multi-tool queries
- Different query types
- Error handling
- Result aggregation

## Best Practices

### Query Formulation

✅ **Good Queries:**
- "What wireless headphones do we have under $200?"
- "Compare our laptop prices with market rates"
- "Which Electronics products have margins below 35%?"
- "Should we adjust pricing for fitness equipment?"

❌ **Avoid:**
- Empty or very vague queries
- Queries requiring external data not available to tools
- Queries mixing unrelated topics

### Tool Selection

The agent automatically selects tools, but understanding the logic helps:

- **Catalog queries**: Use specific product names, categories, or attributes
- **Market queries**: Mention competitors, market prices, or trends
- **Price queries**: Include margin, markup, profit, or pricing keywords
- **Decision queries**: Use "should we", "recommend", or "adjust"

### Error Recovery

```python
try:
    result = agent.process_query(query)
    if result["metadata"]["success"]:
        # Process successful result
        print(result["final_answer"])
    else:
        # Handle error
        print(f"Error: {result['metadata']['error']}")
except ValueError as e:
    # Handle validation errors (empty query, etc.)
    print(f"Invalid query: {e}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

## Limitations

1. **Sequential execution only**: Tools run one after another (no parallel execution yet)
2. **LLM dependency**: Best results require Azure OpenAI credentials (API key + deployment)
3. **Tool limitations**: Inherits limitations from individual tools
4. **Context window**: Very long results may be truncated in aggregation
5. **No conversation history**: Each query is independent

## Future Enhancements

- [ ] Parallel tool execution for independent queries
- [ ] Conversation history and context
- [ ] Custom tool registration
- [ ] Result caching
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Advanced query planning with dependency graphs

## Troubleshooting

### Issue: Agent always uses rule-based routing

**Solution**: Ensure Azure OpenAI environment variables are set, for example:

```bash
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
# or add to .env file
```

### Issue: Tools not executing

**Solution**: Check individual tool initialization and logs

```python
stats = agent.get_stats()
print(stats)  # Check tool availability
```

### Issue: Poor tool selection

**Solution**: 
1. Check query formulation (be specific)
2. Lower temperature for more deterministic routing
3. Check LLM availability (rule-based is less accurate)

### Issue: Slow execution

**Solution**:
1. Use rule-based routing for simple queries
2. Pre-initialize tools
3. Check network connectivity for web search
4. Reduce number of results requested

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify environment variables are set
3. Test individual tools separately
4. Review example queries in `examples/run_agent.py`

---

