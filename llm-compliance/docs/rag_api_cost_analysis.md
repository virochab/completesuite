# RAG API Cost Analysis

## Overview
This document identifies the factors causing API costs in your RAG (Retrieval Augmented Generation) application.

## Primary Cost Drivers

### 1. **LLM API Calls (Main Cost Driver)**
**Location:** `llm-compliance/app/newcbseTest.py:259`

```python
llm=ChatOpenAI(temperature=0, model="gpt-4.1-nano")
```

**Cost Factors:**
- **Every query triggers an LLM call**: Each `/ask` or `/query` request calls the OpenAI API
- **Model**: `gpt-4.1-nano` (check current pricing)
- **Context size**: Includes:
  - System prompt
  - Retrieved context chunks (top 5 documents)
  - User question
  - Generated answer

**Cost Calculation:**
- Input tokens: ~500-2000 tokens per request (depends on retrieved context)
- Output tokens: ~100-500 tokens per answer
- **Cost per request**: Input cost + Output cost

### 2. **No Caching Mechanism**
**Issue:** Identical queries are processed multiple times, each triggering a new API call.

**Current Behavior:**
- Same question asked twice = 2 API calls
- Load testing with repeated questions = many duplicate API calls

**Cost Impact:** High - especially during load testing

### 3. **Retrieval Context Size**
**Location:** `llm-compliance/app/newcbseTest.py:263`

```python
search_kwargs={'k': 5}  # Retrieve top 5 most relevant chunks
```

**Cost Impact:**
- More retrieved chunks = larger context = more input tokens
- Each chunk can be 200-500 tokens
- 5 chunks = ~1000-2500 tokens just for context

### 4. **Load Testing & Concurrent Requests**
**Location:** `llm-compliance/tests/performance/locustLoad.py`

**Cost Factors:**
- Multiple concurrent users (e.g., 50-100 users)
- Each user makes requests every 1-3 seconds
- No rate limiting on API calls
- **Result**: Hundreds of API calls per minute during load tests

### 5. **No Request Deduplication**
**Issue:** Similar questions trigger separate API calls even if answers would be identical.

**Example:**
- "What is the main topic?" 
- "What is the main topic of the document?"
- Both trigger separate API calls

### 6. **Long Timeout Windows**
**Location:** `llm-compliance/app/fastapiragpdfagent.py:174`

```python
max_processing_time = 280  # ~4.7 minutes
```

**Cost Impact:**
- Failed/timeout requests still consume API tokens
- Long timeouts allow expensive requests to run longer

### 7. **Embedding Costs (If Using OpenAI Embeddings)**
**Current Status:** Using local embeddings (`all-mpnet-base-v2`) - **NO COST**

**Note:** If switched to OpenAI embeddings, this would add:
- ~$0.0001 per 1K tokens for embeddings
- Called during vector store creation and potentially during retrieval

## Cost Breakdown Example

### Single Request Cost:
```
Input tokens:  ~1,500 tokens (prompt + context)
Output tokens: ~200 tokens (answer)
Total:         ~1,700 tokens

Cost per request (example with gpt-4.1-nano):
- Input:  1,500 tokens × $X per 1K tokens
- Output: 200 tokens × $Y per 1K tokens
- Total:  ~$Z per request
```

### Load Test Cost (50 users, 5 minutes):
```
Requests per user: ~100 requests (1 request every 3 seconds)
Total requests: 50 users × 100 = 5,000 requests
Cost: 5,000 × $Z = $XXX
```

## Cost Optimization Recommendations

### 1. **Implement Response Caching** ⭐ HIGH IMPACT
```python
# Add caching for identical queries
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_qa_chain(query_hash, query):
    return agent.qa_chain.invoke({"query": query})
```

**Expected Savings:** 30-50% reduction for repeated queries

### 2. **Reduce Retrieval Context Size**
```python
search_kwargs={'k': 3}  # Reduce from 5 to 3
```

**Expected Savings:** 20-30% reduction in input tokens

### 3. **Add Request Rate Limiting**
```python
# Limit concurrent API calls
from asyncio import Semaphore
api_semaphore = Semaphore(10)  # Max 10 concurrent API calls
```

**Expected Savings:** Prevents cost spikes during load tests

### 4. **Use Cheaper Model for Simple Queries**
```python
# Use cheaper model for simple questions
if is_simple_query(question):
    model = "gpt-3.5-turbo"  # Cheaper
else:
    model = "gpt-4.1-nano"  # More capable
```

**Expected Savings:** 50-70% cost reduction for simple queries

### 5. **Implement Request Deduplication**
```python
# Cache similar queries
similarity_threshold = 0.95
cached_queries = {}

def get_cached_answer(query):
    for cached_query, answer in cached_queries.items():
        if similarity(query, cached_query) > similarity_threshold:
            return answer
    return None
```

**Expected Savings:** 10-20% reduction

### 6. **Add Max Tokens Limit**
```python
llm=ChatOpenAI(
    temperature=0, 
    model="gpt-4.1-nano",
    max_tokens=500  # Limit response length
)
```

**Expected Savings:** 20-30% reduction in output token costs

### 7. **Monitor and Log API Usage**
```python
# Track API usage
def track_api_usage(query, response):
    input_tokens = count_tokens(query + context)
    output_tokens = count_tokens(response)
    log_api_cost(input_tokens, output_tokens)
```

**Benefit:** Visibility into actual costs

### 8. **Optimize Prompts**
- Shorter prompts = fewer input tokens
- More specific prompts = better answers with fewer tokens

**Expected Savings:** 10-15% reduction

## Immediate Actions

1. **Add caching** - Biggest impact, easiest to implement
2. **Reduce k from 5 to 3** - Quick win, minimal quality impact
3. **Add max_tokens limit** - Prevents runaway costs
4. **Monitor API usage** - Understand actual costs

## Cost Monitoring

### Check OpenAI Usage Dashboard
1. Go to https://platform.openai.com/usage
2. Filter by model: `gpt-4.1-nano`
3. Check token usage and costs

### Add Cost Tracking to Your App
```python
# Add to fastapiragpdfagent.py
import tiktoken

def estimate_cost(input_text, output_text, model="gpt-4.1-nano"):
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(input_text))
    output_tokens = len(encoding.encode(output_text))
    
    # Check current pricing at https://openai.com/pricing
    input_cost_per_1k = 0.001  # Update with actual pricing
    output_cost_per_1k = 0.002  # Update with actual pricing
    
    cost = (input_tokens / 1000 * input_cost_per_1k) + \
           (output_tokens / 1000 * output_cost_per_1k)
    return cost, input_tokens, output_tokens
```

## Summary

**Main Cost Drivers:**
1. LLM API calls (every query) - **PRIMARY**
2. No caching (duplicate queries)
3. Large context windows (5 retrieved chunks)
4. Load testing (many concurrent requests)
5. No rate limiting

**Quick Wins:**
- Implement caching (30-50% savings)
- Reduce k from 5 to 3 (20-30% savings)
- Add max_tokens limit (20-30% savings)
- **Total potential savings: 50-70%**

