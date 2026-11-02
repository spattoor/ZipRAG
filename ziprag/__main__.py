import argparse
import json
import os
from pathlib import Path
import shutil
from .service import ROOT, ZipRAG
from .retrieval import build_index, build_openai_embeddings


def main():
    parser = argparse.ArgumentParser(description='ZipRAG: free document retrieval and ZIP lookup')
    parser.add_argument('command', choices=['build', 'search', 'scores', 'evaluate', 'sync-scores', 'openai-embeddings'])
    parser.add_argument('query', nargs='?', default='')
    args = parser.parse_args()
    if args.command == 'build':
        result = build_index(ROOT / 'docs', ROOT / 'data/documents.db')
    elif args.command == 'sync-scores':
        from .snapshot import restore
        import tempfile
        path = restore(Path(tempfile.gettempdir()) / 'ziprag-snapshot-cache')
        shutil.copyfile(path, ROOT / 'data/austin.db')
        result = ZipRAG().scores.metadata()
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate()
    elif args.command == 'search':
        result = ZipRAG().search_documents(args.query)
    elif args.command == 'scores':
        result = ZipRAG().lookup_zip_scores(args.query.split(','))
    else:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
        if not os.environ.get('OPENAI_API_KEY'):
            raise SystemExit('Set OPENAI_API_KEY locally to opt into provider-billed embeddings. Free search needs no key.')
        from openai import OpenAI
        database = ROOT / 'data/documents.db'
        if not database.exists():
            build_index(ROOT / 'docs', database)
        result = {'embedded_chunks': build_openai_embeddings(database, OpenAI(max_retries=0, timeout=30))}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
