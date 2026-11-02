"""Source-level hit@3 on ten authored questions; not an answer-quality benchmark."""
from datetime import datetime, timezone
import json
from .service import ROOT, ZipRAG


def evaluate():
    cases = json.loads((ROOT / 'evaluation/questions.json').read_text())
    service = ZipRAG()
    rows = []
    for case in cases:
        results = service.search_documents(case['question'])['results']
        retrieved = [r['url'] for r in results]
        rows.append(dict(case, retrieved_sources=retrieved, hit=case['expected_source'] in retrieved))
    hits = sum(r['hit'] for r in rows)
    result = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'questions': len(rows),
              'hits_at_3': hits, 'hit_at_3': hits / len(rows),
              'corpus': service.documents.metadata(), 'cases': rows,
              'limitations': 'Small development set authored for this implementation. Measures article retrieval, not answer correctness, generalization, or Claude performance. No model API was called.'}
    (ROOT / 'evaluation/results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
