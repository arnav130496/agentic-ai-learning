CHUNKS = {
    "c1": {"chunk_id": "c1", "text": "Employees accrue 1.5 days of PTO per month, capped at 18 days annually.", "source": "hr_policy_pto.md"},
    "c2": {"chunk_id": "c2", "text": "Unused PTO up to 5 days may be carried over into the next calendar year.", "source": "hr_policy_pto.md"},
    "c3": {"chunk_id": "c3", "text": "Remote work requests must be submitted to a manager at least 2 weeks in advance.", "source": "hr_policy_remote.md"},
    "c4": {"chunk_id": "c4", "text": "Expense reports must be filed within 30 days of purchase to be reimbursed.", "source": "hr_policy_expenses.md"},
    "c5": {"chunk_id": "c5", "text": "The standard reimbursement method is direct deposit within 10 business days of approval.", "source": "hr_policy_expenses.md"},
    "c6": {"chunk_id": "c6", "text": "New hires complete a 90-day probationary period before benefits fully vest.", "source": "hr_policy_onboarding.md"},
}

TEST_QUERIES = [
    {
        # Case A: strong evidence — should clear the gate
        "query": "How much PTO do employees accrue per month?",
        "retrieved_chunks": [
            {**CHUNKS["c1"], "rerank_score": 0.91},
            {**CHUNKS["c2"], "rerank_score": 0.74},
            {**CHUNKS["c6"], "rerank_score": 0.22},
        ],
    },
    {
        # Case B: no relevant evidence at all — should trigger insufficient_evidence
        "query": "What is the company's policy on parental leave?",
        "retrieved_chunks": [
            {**CHUNKS["c3"], "rerank_score": 0.31},
            {**CHUNKS["c6"], "rerank_score": 0.28},
            {**CHUNKS["c4"], "rerank_score": 0.19},
        ],
    },
    {
        # Case C: borderline — use this to pressure-test wherever you set the threshold
        "query": "How long do I have to submit an expense report?",
        "retrieved_chunks": [
            {**CHUNKS["c4"], "rerank_score": 0.52},
            {**CHUNKS["c5"], "rerank_score": 0.48},
            {**CHUNKS["c3"], "rerank_score": 0.15},
        ],
    }
]