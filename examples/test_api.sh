#!/bin/bash

# API Testing Script for AI Product Research Assistant
# This script tests all API endpoints with various scenarios

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# API base URL
API_URL="http://localhost:8000"

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Function to print test descriptions
print_test() {
    echo -e "${YELLOW}Test: $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}\n"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}\n"
}

# Check if server is running
print_header "Checking Server Status"
print_test "Checking if API server is running at $API_URL"

if curl -s "$API_URL/health" > /dev/null; then
    print_success "Server is running"
else
    print_error "Server is not running. Please start it with: uvicorn src.api.main:app --reload"
    exit 1
fi

# Test 1: Health Check
print_header "Test 1: Health Check Endpoint"
print_test "GET /health - Check system health"

echo "Request:"
echo "curl -X GET \"$API_URL/health\""
echo ""

echo "Response:"
curl -s -X GET "$API_URL/health" | python -m json.tool

print_success "Health check completed"

# Test 2: Root Endpoint
print_header "Test 2: Root Endpoint"
print_test "GET / - Get API information"

echo "Request:"
echo "curl -X GET \"$API_URL/\""
echo ""

echo "Response:"
curl -s -X GET "$API_URL/" | python -m json.tool

print_success "Root endpoint test completed"

# Test 3: Product Catalog Query
print_header "Test 3: Product Catalog Query"
print_test "POST /query - Search for wireless headphones"

QUERY_1='{"query": "What wireless headphones do we have in stock?"}'

echo "Request:"
echo "curl -X POST \"$API_URL/query\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$QUERY_1'"
echo ""

echo "Response:"
RESPONSE_1=$(curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_1")

echo "$RESPONSE_1" | python -m json.tool

# Extract query_id for later use
QUERY_ID_1=$(echo "$RESPONSE_1" | python -c "import sys, json; print(json.load(sys.stdin).get('query_id', ''))" 2>/dev/null)

if [ -n "$QUERY_ID_1" ]; then
    print_success "Query processed successfully. Query ID: $QUERY_ID_1"
else
    print_error "Failed to process query"
fi

# Test 4: Price Analysis Query
print_header "Test 4: Price Analysis Query"
print_test "POST /query - Analyze profit margins"

QUERY_2='{"query": "Which products have profit margins below 40%?"}'

echo "Request:"
echo "curl -X POST \"$API_URL/query\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$QUERY_2'"
echo ""

echo "Response:"
RESPONSE_2=$(curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_2")

echo "$RESPONSE_2" | python -m json.tool

QUERY_ID_2=$(echo "$RESPONSE_2" | python -c "import sys, json; print(json.load(sys.stdin).get('query_id', ''))" 2>/dev/null)

if [ -n "$QUERY_ID_2" ]; then
    print_success "Query processed successfully. Query ID: $QUERY_ID_2"
else
    print_error "Failed to process query"
fi

# Test 5: Web Search Query
print_header "Test 5: Web Search Query"
print_test "POST /query - Search for market prices"

QUERY_3='{"query": "What is the current market price for Sony WH-1000XM5 headphones?"}'

echo "Request:"
echo "curl -X POST \"$API_URL/query\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$QUERY_3'"
echo ""

echo "Response:"
RESPONSE_3=$(curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_3")

echo "$RESPONSE_3" | python -m json.tool

QUERY_ID_3=$(echo "$RESPONSE_3" | python -c "import sys, json; print(json.load(sys.stdin).get('query_id', ''))" 2>/dev/null)

if [ -n "$QUERY_ID_3" ]; then
    print_success "Query processed successfully. Query ID: $QUERY_ID_3"
else
    print_error "Failed to process query"
fi

# Test 6: Multi-Tool Query
print_header "Test 6: Multi-Tool Query (Comprehensive Analysis)"
print_test "POST /query - Pricing decision with multiple tools"

QUERY_4='{"query": "Should we lower the price of AudioMax Pro headphones based on competitor pricing?"}'

echo "Request:"
echo "curl -X POST \"$API_URL/query\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$QUERY_4'"
echo ""

echo "Response:"
RESPONSE_4=$(curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_4")

echo "$RESPONSE_4" | python -m json.tool

QUERY_ID_4=$(echo "$RESPONSE_4" | python -c "import sys, json; print(json.load(sys.stdin).get('query_id', ''))" 2>/dev/null)

if [ -n "$QUERY_ID_4" ]; then
    print_success "Query processed successfully. Query ID: $QUERY_ID_4"
else
    print_error "Failed to process query"
fi

# Test 7: Get Query History (All)
print_header "Test 7: Get Query History (All Queries)"
print_test "GET /queries - Retrieve all queries"

echo "Request:"
echo "curl -X GET \"$API_URL/queries\""
echo ""

echo "Response:"
curl -s -X GET "$API_URL/queries" | python -m json.tool

print_success "Query history retrieved"

# Test 8: Get Query History (Paginated)
print_header "Test 8: Get Query History (Paginated)"
print_test "GET /queries?limit=2&offset=0 - Get first 2 queries"

echo "Request:"
echo "curl -X GET \"$API_URL/queries?limit=2&offset=0\""
echo ""

echo "Response:"
curl -s -X GET "$API_URL/queries?limit=2&offset=0" | python -m json.tool

print_success "Paginated query history retrieved"

# Test 9: Submit Feedback (Positive)
print_header "Test 9: Submit Feedback (Positive)"
print_test "POST /feedback - Submit positive feedback"

if [ -n "$QUERY_ID_1" ]; then
    FEEDBACK_1="{\"query_id\": \"$QUERY_ID_1\", \"rating\": 5, \"comment\": \"Very helpful! Found exactly what I was looking for.\"}"
    
    echo "Request:"
    echo "curl -X POST \"$API_URL/feedback\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '$FEEDBACK_1'"
    echo ""
    
    echo "Response:"
    curl -s -X POST "$API_URL/feedback" \
      -H "Content-Type: application/json" \
      -d "$FEEDBACK_1" | python -m json.tool
    
    print_success "Positive feedback submitted"
else
    print_error "No query ID available for feedback"
fi

# Test 10: Submit Feedback (Negative)
print_header "Test 10: Submit Feedback (Negative)"
print_test "POST /feedback - Submit negative feedback"

if [ -n "$QUERY_ID_2" ]; then
    FEEDBACK_2="{\"query_id\": \"$QUERY_ID_2\", \"rating\": 2, \"comment\": \"Results were not very accurate.\"}"
    
    echo "Request:"
    echo "curl -X POST \"$API_URL/feedback\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '$FEEDBACK_2'"
    echo ""
    
    echo "Response:"
    curl -s -X POST "$API_URL/feedback" \
      -H "Content-Type: application/json" \
      -d "$FEEDBACK_2" | python -m json.tool
    
    print_success "Negative feedback submitted"
else
    print_error "No query ID available for feedback"
fi

# Test 11: Submit Feedback (Without Comment)
print_header "Test 11: Submit Feedback (Without Comment)"
print_test "POST /feedback - Submit feedback without comment"

if [ -n "$QUERY_ID_3" ]; then
    FEEDBACK_3="{\"query_id\": \"$QUERY_ID_3\", \"rating\": 4}"
    
    echo "Request:"
    echo "curl -X POST \"$API_URL/feedback\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '$FEEDBACK_3'"
    echo ""
    
    echo "Response:"
    curl -s -X POST "$API_URL/feedback" \
      -H "Content-Type: application/json" \
      -d "$FEEDBACK_3" | python -m json.tool
    
    print_success "Feedback without comment submitted"
else
    print_error "No query ID available for feedback"
fi

# Test 12: Error Handling - Empty Query
print_header "Test 12: Error Handling - Empty Query"
print_test "POST /query - Test with empty query"

QUERY_EMPTY='{"query": ""}'

echo "Request:"
echo "curl -X POST \"$API_URL/query\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$QUERY_EMPTY'"
echo ""

echo "Response:"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_EMPTY" | python -m json.tool

print_success "Error handling test completed"

# Test 13: Error Handling - Invalid Feedback Rating
print_header "Test 13: Error Handling - Invalid Feedback Rating"
print_test "POST /feedback - Test with invalid rating (out of range)"

FEEDBACK_INVALID='{"query_id": "test123", "rating": 10, "comment": "Invalid rating"}'

echo "Request:"
echo "curl -X POST \"$API_URL/feedback\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$FEEDBACK_INVALID'"
echo ""

echo "Response:"
curl -s -X POST "$API_URL/feedback" \
  -H "Content-Type: application/json" \
  -d "$FEEDBACK_INVALID" | python -m json.tool

print_success "Error handling test completed"

# Test 14: Error Handling - Non-existent Query ID
print_header "Test 14: Error Handling - Non-existent Query ID"
print_test "POST /feedback - Test with non-existent query ID"

FEEDBACK_NOTFOUND='{"query_id": "nonexistent123", "rating": 5, "comment": "This should fail"}'

echo "Request:"
echo "curl -X POST \"$API_URL/feedback\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '$FEEDBACK_NOTFOUND'"
echo ""

echo "Response:"
curl -s -X POST "$API_URL/feedback" \
  -H "Content-Type: application/json" \
  -d "$FEEDBACK_NOTFOUND" | python -m json.tool

print_success "Error handling test completed"

# Summary
print_header "Test Summary"
echo -e "${GREEN}All API endpoint tests completed!${NC}"
echo ""
echo "Tested endpoints:"
echo "  ✓ GET  /          - Root endpoint"
echo "  ✓ GET  /health    - Health check"
echo "  ✓ POST /query     - Process queries (4 different types)"
echo "  ✓ GET  /queries   - Retrieve query history (with and without pagination)"
echo "  ✓ POST /feedback  - Submit feedback (3 different scenarios)"
echo ""
echo "Error handling tests:"
echo "  ✓ Empty query"
echo "  ✓ Invalid rating"
echo "  ✓ Non-existent query ID"
echo ""
echo -e "${BLUE}For interactive API documentation, visit:${NC}"
echo "  - Swagger UI: $API_URL/docs"
echo "  - ReDoc:      $API_URL/redoc"
echo ""

