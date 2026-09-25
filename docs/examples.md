# Examples

## Basic Usage

```python
research("What is retrieval augmented generation?")
# → Domain: general, Engine: google
```

## News

```python
research("Latest AI news")
# → Domain: news, Engine: google_news
```

## Academic Research

```python
research("Research papers about RAG hallucination")
# → Domain: academic, Engine: google_scholar
```

## Jobs

```python
research("AI internships in Chennai posted this week")
# → Domain: jobs, Engine: google_jobs
# → Location: Chennai, Date filter: this week
```

## Shopping

```python
research("RTX laptops under ₹90000 for machine learning")
# → Domain: shopping, Engine: google_shopping
# → Price filter: max ₹90,000
```

## Places

```python
research("Best cafes near Chennai airport")
# → Domain: places, Engine: google_maps
```

## Deep Research

```python
research("Research recent RAG hallucination mitigation techniques", depth="deep")
# → Multiple searches executed concurrently
# → Results aggregated, deduplicated, and ranked
```

## Debug / Explain

```python
explain_research_plan("Find AI internships in Chennai posted this week")
# Returns:
# {
#   "domain": "jobs",
#   "confidence": 0.85,
#   "entities": {"location": "Chennai", "date_range": "this_week"},
#   "engine": "google_jobs",
#   "reason": "Jobs intent detected with 85% confidence"
# }
```
