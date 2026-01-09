import giskard

# Define your training/reference data
reference_data = giskard.Dataset(
    df=historical_financial_queries,
    target="query_type"
)

# Test for drift in production data
production_data = giskard.Dataset(
    df=current_queries,
    target="query_type"
)

# Detect if query patterns have changed
drift_test = giskard.testing.test_drift(
    model=rag_model,
    dataset=production_data,
    reference_dataset=reference_data
)

# Example: Are users suddenly asking different types of questions?
# Training: 80% investment queries, 20% account queries
# Production: 60% investment, 40% crypto queries <- DRIFT DETECTED