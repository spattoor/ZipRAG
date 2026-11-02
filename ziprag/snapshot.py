"""Bounded HTTPS restore from the Analyzer's public CloudFront snapshot."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.request import Request, build_opener, HTTPRedirectHandler

BASE_URL = 'https://d2oesjmed1xezr.cloudfront.net'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_url(path, maximum, opener=None):
    if path != 'latest.json' and not re.fullmatch(r'snapshots/[a-f0-9]{64}\.db', path):
        raise ValueError('Invalid published snapshot path.')
    opener = opener or build_opener(NoRedirect)
    request = Request(BASE_URL + '/' + path, headers={'User-Agent': 'ZipRAG/1.0 portfolio snapshot reader'})
    with opener.open(request, timeout=20) as response:
        if response.status != 200:
            raise ValueError('Snapshot request did not return HTTP 200.')
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Snapshot response exceeded its size limit.')
    return data


def restore(directory, reader=read_url):
    manifest = json.loads(reader('latest.json', 16384))
    checksum = manifest.get('sha256')
    if (manifest.get('schema_version') != 1 or not isinstance(checksum, str)
            or not re.fullmatch(r'[a-f0-9]{64}', checksum)
            or manifest.get('database_key') != f'snapshots/{checksum}.db'):
        raise ValueError('Invalid snapshot manifest.')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (checksum + '.db')
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
        return target
    payload = reader(manifest['database_key'], 50 * 1024 * 1024)
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError('Snapshot checksum mismatch.')
    fd, temp = tempfile.mkstemp(dir=directory, suffix='.pending')
    os.close(fd)
    try:
        Path(temp).write_bytes(payload)
        with closing(sqlite3.connect(Path(temp).as_uri() + '?mode=ro', uri=True)) as conn:
            if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Snapshot failed SQLite integrity checks.')
            conn.execute('SELECT zip_code,observed_on,comparison_score FROM market_observations LIMIT 1')
            conn.execute('SELECT json FROM snapshot_metadata LIMIT 1')
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)
    return target
