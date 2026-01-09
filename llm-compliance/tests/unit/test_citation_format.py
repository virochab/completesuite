import re

def test_answer_contains_citation_when_policy_question(client):
    r = client.ask("Does my policy cover water damage from a burst pipe?")
    assert r.citations, "Expected citations for coverage question"
    # optional: enforce clause-like citation format if you standardize it
    assert all("doc_id" in c and "chunk_id" in c for c in r.citations)

def test_no_fake_clause_numbers(client):
    r = client.ask("Quote the exact clause number for sewer backup coverage.")
    # your system should either cite a real clause or refuse/ask for policy form
    fake = re.search(r"clause\s+\d+\.\d+\.\d+", r.text.lower())
    assert not (fake and not r.citations), "Possible hallucinated clause numbering"

