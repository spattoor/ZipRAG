"""Local TF-IDF vectors + SQLite FTS5. No model, key or network required."""
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import os

STOP = set("a an and are as at be by can did do does for from had has have how i in is it its me of on or our that the their there this to was were what when where which who why will with would about tell give please you your".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOP]


def query_text(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 600:
        raise ValueError("Enter a question between 1 and 600 characters.")
    return query.strip()


def chunks(text, size=180, overlap=35):
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def vector(text, idf):
    counts = Counter(tokens(text))
    weighted = {t: (1 + math.log(n)) * idf[t] for t, n in counts.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in weighted.values())) or 1
    return {t: v / norm for t, v in weighted.items()}


def build_index(docs, destination):
    docs, destination = Path(docs), Path(destination)
    manifest_bytes = (docs / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = []
    for item in manifest:
        filename = item["file"]
        if Path(filename).name != filename:
            raise ValueError("Document path must stay inside the corpus.")
        payload = (docs / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Document checksum mismatch: {filename}")
        for ordinal, text in enumerate(chunks(payload.decode("utf-8"))):
            rows.append((f"D{filename.removesuffix('.txt')}-{ordinal}", item, text))
    if not rows:
        raise ValueError("Corpus is empty.")
    frequencies = Counter(t for _, item, text in rows for t in set(tokens(item['title'] + ' ' + text)))
    idf = {t: math.log((1 + len(rows)) / (1 + n)) + 1 for t, n in frequencies.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".index-", suffix=".db", dir=destination.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            conn.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks(id TEXT PRIMARY KEY, title TEXT, text TEXT, url TEXT,
                    revision INTEGER, retrieved_at TEXT, license TEXT, vector TEXT);
                CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, title, text);
                CREATE TABLE embeddings(id TEXT PRIMARY KEY, model TEXT, vector TEXT);
            """)
            for cid, item, text in rows:
                conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)", (
                    cid, item['title'], text, item['url'], item['revision'], item['retrieved_at'],
                    item['license'], json.dumps(vector(item['title'] + ' ' + text, idf))))
                conn.execute("INSERT INTO search VALUES (?,?,?)", (cid, item['title'], text))
            meta = {"idf": idf, "documents": len(manifest), "chunks": len(rows),
                    "corpus_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "retrieval": "SQLite FTS5 + local TF-IDF cosine similarity"}
            conn.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
            conn.commit()
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {k: v for k, v in meta.items() if k != 'idf'}


class DocumentSearch:
    def __init__(self, database):
        self.database = Path(database).resolve()

    def connect(self):
        conn = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def metadata(self):
        with closing(self.connect()) as conn:
            return {r['key']: json.loads(r['value']) for r in conn.execute("SELECT * FROM metadata WHERE key != 'idf'")}

    def search(self, query, limit=3, embedding_client=None):
        query = query_text(query)
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Result limit must be between 1 and 5.")
        terms = tokens(query)
        if not terms:
            return []
        with closing(self.connect()) as conn:
            idf = json.loads(conn.execute("SELECT value FROM metadata WHERE key='idf'").fetchone()[0])
            qvec = vector(query, idf)
            rows = list(conn.execute("SELECT * FROM chunks"))
            lexical = {r['id']: r['rank'] for r in conn.execute(
                "SELECT id,bm25(search,0,4,1) AS rank FROM search WHERE search MATCH ? ORDER BY rank LIMIT 40",
                (' OR '.join('"' + t + '"' for t in dict.fromkeys(terms)),))}
            sparse = sorted(((sum(qvec.get(t, 0) * v for t, v in json.loads(r['vector']).items()), r['id'])
                             for r in rows), reverse=True)
            sparse_rank = {cid: n for n, (score, cid) in enumerate(sparse, 1) if score > 0}
            lexical_rank = {cid: n for n, cid in enumerate(sorted(lexical, key=lexical.get), 1)}
            dense_rank = {}
            if embedding_client is not None:
                embedded = list(conn.execute("SELECT * FROM embeddings"))
                if not embedded:
                    raise ValueError("Build the optional OpenAI embedding index first.")
                model = embedded[0]['model']
                qdense = embedding_client.embeddings.create(input=[query], model=model).data[0].embedding
                norm = math.sqrt(sum(x*x for x in qdense)) or 1
                ranked = []
                for r in embedded:
                    v = json.loads(r['vector'])
                    if len(v) != len(qdense):
                        raise ValueError("Embedding dimensions do not match.")
                    vn = math.sqrt(sum(x*x for x in v)) or 1
                    ranked.append((sum(x*y for x,y in zip(v, qdense)) / (norm*vn), r['id']))
                dense_rank = {cid: n for n, (_, cid) in enumerate(sorted(ranked, reverse=True), 1)}
            scored = []
            for r in rows:
                cid = r['id']
                if cid not in sparse_rank and cid not in lexical_rank and cid not in dense_rank:
                    continue
                score = sum(1/(40 + ranks[cid]) for ranks in (sparse_rank, lexical_rank, dense_rank) if cid in ranks)
                scored.append((score, dict(r)))
        results = []
        for score, row in sorted(scored, key=lambda x: (-x[0], x[1]['id'])):
            # Evaluate source retrieval, not multiple chunks from the same article.
            if any(r['url'] == row['url'] for r in results):
                continue
            row.pop('vector')
            row.update(citation_id=row.pop('id'), retrieval_score=round(score, 6))
            results.append(row)
            if len(results) == limit:
                break
        return results


def build_openai_embeddings(database, client, model="text-embedding-3-small"):
    """Explicit opt-in: incurs provider usage. Baseline index remains usable."""
    with closing(sqlite3.connect(database)) as conn:
        rows = conn.execute("SELECT id,title,text FROM chunks ORDER BY id").fetchall()
        response = client.embeddings.create(model=model, input=[title + '\n' + text for _, title, text in rows])
        if len(response.data) != len(rows):
            raise ValueError("Incomplete embedding response.")
        indexed = sorted(response.data, key=lambda x: x.index)
        with conn:
            conn.execute("DELETE FROM embeddings")
            conn.executemany("INSERT INTO embeddings VALUES (?,?,?)", [
                (row[0], model, json.dumps(result.embedding)) for row, result in zip(rows, indexed)])
    return len(rows)
