import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from ziprag.service import ROOT, ZipRAG
from ziprag.retrieval import build_index, DocumentSearch, build_openai_embeddings
from ziprag.claude import answer, dispatch
from ziprag.snapshot import restore


class ZipRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ZipRAG()

    def test_article_retrieval_cites_real_source_and_revision(self):
        rows = self.engine.search_documents('Which neighborhood contains Ramsey Park?')['results']
        self.assertIn('Rosedale', rows[0]['title'])
        self.assertTrue(rows[0]['citation_id'].startswith('D'))
        self.assertIsInstance(rows[0]['revision'], int)
        self.assertEqual(len({r['url'] for r in rows}), len(rows))

    def test_unknown_terms_have_no_fabricated_results(self):
        self.assertEqual(self.engine.search_documents('xyzzyunsupportedtoken')['results'], [])

    def test_bounded_search_and_malicious_fts_syntax(self):
        for query in ['', 'x'*601]:
            with self.assertRaises(ValueError): self.engine.search_documents(query)
        with self.assertRaises(ValueError): self.engine.search_documents('Austin', 100)
        self.assertIsInstance(self.engine.search_documents('" OR * ; DROP TABLE chunks; --')['results'], list)
        self.assertEqual(self.engine.documents.metadata()['documents'], 18)

    def test_checksum_tampering_refused_before_index_creation(self):
        with tempfile.TemporaryDirectory() as td:
            docs = Path(td) / 'docs'
            shutil.copytree(ROOT / 'docs', docs)
            record = json.loads((docs / 'manifest.json').read_text())[0]
            (docs / record['file']).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                build_index(docs, Path(td) / 'index.db')
            self.assertFalse((Path(td) / 'index.db').exists())

    def test_zip_lookup_preserves_exact_month_and_provenance(self):
        result = self.engine.lookup_zip_scores(['78701','78704'])
        self.assertEqual(len(result['records']), 2)
        self.assertTrue(all(r['observed_on'] == result['month'] for r in result['records']))
        self.assertTrue(all(r['citation_id'].startswith('Z') for r in result['records']))
        self.assertIn('zillowstatic', result['source_url'])
        self.assertEqual(self.engine.scores.metadata()['zip_count'], 42)

    def test_missing_zip_and_date_are_explicit(self):
        result = self.engine.lookup_zip_scores(['00000'])
        self.assertEqual(result['records'], [])
        self.assertEqual(result['missing_zip_codes'], ['00000'])
        self.assertEqual(self.engine.lookup_zip_scores(['78701'],'1900-01-01')['records'], [])
        with self.assertRaises(ValueError): self.engine.lookup_zip_scores(['78701'], '2026-99-99')

    def test_zip_sql_injection_and_excessive_input_refused(self):
        for codes in [["78701'); DROP TABLE market_observations;--"], ['78701']*6, [], '78701']:
            with self.assertRaises(ValueError): self.engine.lookup_zip_scores(codes)
        self.assertEqual(self.engine.scores.metadata()['zip_count'], 42)

    def test_no_automatic_neighborhood_zip_mapping(self):
        self.assertIsNone(self.engine.evidence('Tell me about Hyde Park')['scores'])
        with self.assertRaises(ValueError): self.engine.evidence('78701 78702 78703 78704 78705 78706')

    def test_snapshot_restore_checks_checksum_and_reuses_valid_cache(self):
        payload = (ROOT / 'data/austin.db').read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        calls = []
        def reader(key, maximum):
            calls.append(key)
            return json.dumps({'schema_version':1,'sha256':sha,'database_key':f'snapshots/{sha}.db'}).encode() if key == 'latest.json' else payload
        with tempfile.TemporaryDirectory() as td:
            target = restore(td,reader)
            self.assertEqual(target.read_bytes(),payload)
            restore(td,reader)
            self.assertEqual(len(calls),3)

    def test_corrupted_snapshot_never_published(self):
        def reader(key, maximum):
            sha = 'a'*64
            return json.dumps({'schema_version':1,'sha256':sha,'database_key':f'snapshots/{sha}.db'}).encode() if key == 'latest.json' else b'not a database'
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'checksum'): restore(td,reader)
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_tool_dispatch_rejects_unknown_tools_and_extra_parameters(self):
        with self.assertRaises(ValueError): dispatch(self.engine,'execute_sql',{'query':'SELECT *'})
        with self.assertRaises(ValueError): dispatch(self.engine,'search_documents',{'query':'Austin','file':'/etc/passwd'})

    def client(self, invalid=False):
        def create(**kwargs):
            last = kwargs['messages'][-1]
            if last['role'] == 'user' and isinstance(last['content'],str):
                return SimpleNamespace(content=[{'type':'tool_use','id':'tool-1','name':'search_documents','input':{'query':'Hyde Park history'}}],stop_reason='tool_use')
            result = json.loads(last['content'][0]['content'])
            cid = 'D999999-0' if invalid else result['results'][0]['citation_id']
            return SimpleNamespace(content=[{'type':'text','text':f'Here is retrieved evidence. [{cid}]'}],stop_reason='end_turn')
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    def test_claude_tool_loop_with_stub_never_claimed_as_live_model(self):
        result = answer('Hyde Park history',self.engine,self.client(),'test-model-stub')
        self.assertEqual(len(result['trace']),1)
        self.assertEqual(result['trace'][0]['tool'],'search_documents')
        self.assertTrue(result['sources'])

    def test_claude_unknown_citation_rejected(self):
        with self.assertRaisesRegex(ValueError,'citations'):
            answer('Hyde Park history',self.engine,self.client(True),'test-model-stub')

    def test_optional_embeddings_with_stub_preserve_real_source_records(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'index.db'
            build_index(ROOT/'docs',db)
            def create(**kwargs):
                return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[1.0,0.0]) for i,_ in enumerate(kwargs['input'])])
            stub = SimpleNamespace(embeddings=SimpleNamespace(create=create))
            count = build_openai_embeddings(db,stub)
            self.assertGreater(count,18)
            self.assertTrue(DocumentSearch(db).search('Ramsey Park',embedding_client=stub))


if __name__ == '__main__': unittest.main()
