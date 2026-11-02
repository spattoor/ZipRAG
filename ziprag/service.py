"""One service shared by the browser, Claude tool loop and MCP server."""
import os
from pathlib import Path
import re
import tempfile
from .retrieval import DocumentSearch, query_text
from .scores import ScoreLookup

ROOT = Path(__file__).resolve().parents[1]


class ZipRAG:
    def __init__(self, index=None, scores=None, origin='Bundled Analyzer snapshot'):
        if index is None:
            import hashlib
            from .retrieval import build_index
            fingerprint = hashlib.sha256((ROOT / 'docs/manifest.json').read_bytes()).hexdigest()
            index = Path(tempfile.gettempdir()) / f'ziprag-fts-tfidf-v1-{fingerprint}.db'
            if not index.exists():
                build_index(ROOT / 'docs', index)
        self.documents = DocumentSearch(index)
        self.scores = ScoreLookup(scores or os.environ.get('ZIPRAG_SCORE_DB') or ROOT / 'data/austin.db')
        self.origin = origin

    def search_documents(self, query, limit=3):
        return {'results': self.documents.search(query, limit),
                'method': 'SQLite FTS5 + local TF-IDF vectors; no model API call',
                'limitations': 'Wikipedia reference excerpts, not an authoritative neighborhood-to-ZIP mapping.'}

    def lookup_zip_scores(self, zip_codes, month=None):
        return dict(self.scores.lookup(zip_codes, month), snapshot_origin=self.origin)

    def evidence(self, query):
        query = query_text(query)
        found = list(dict.fromkeys(re.findall(r'\b\d{5}\b', query)))
        if len(found) > 5:
            raise ValueError('Compare no more than five ZIP codes at a time.')
        return {'documents': self.search_documents(query),
                'scores': self.lookup_zip_scores(found) if found else None,
                'mode': 'Search and data: source excerpts and stored values, not an AI-generated answer.'}


def live_service():
    """A network failure is explicit; the shipped, dated snapshot stays usable."""
    from .snapshot import restore
    cache = Path(tempfile.gettempdir()) / 'ziprag-snapshot-cache'
    try:
        path = restore(cache)
        return ZipRAG(scores=path, origin='AWS S3 snapshot via CloudFront'), None
    except Exception:
        return ZipRAG(), 'Live snapshot unavailable. Showing the included, dated Analyzer snapshot.'
