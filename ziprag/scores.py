"""Parameterized, read-only queries over the Analyzer's actual dated scores."""
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import re
import sqlite3

METHOD = "40% lower-price percentile + 35% one-year-change percentile + 25% three-year-change percentile"


class ScoreLookup:
    def __init__(self, database):
        self.database = Path(database).resolve()

    def connect(self):
        conn = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def metadata(self):
        with closing(self.connect()) as conn:
            metadata = json.loads(conn.execute('SELECT json FROM snapshot_metadata LIMIT 1').fetchone()[0])
            metadata['zip_count'] = conn.execute('SELECT COUNT(DISTINCT zip_code) FROM market_observations').fetchone()[0]
            metadata['row_count'] = conn.execute('SELECT COUNT(*) FROM market_observations').fetchone()[0]
            metadata['latest_month'] = conn.execute('SELECT MAX(observed_on) FROM market_observations').fetchone()[0]
            return metadata

    def lookup(self, zip_codes, month=None):
        if not isinstance(zip_codes, list) or not 1 <= len(zip_codes) <= 5:
            raise ValueError('Provide between 1 and 5 ZIP codes.')
        if any(not isinstance(z, str) or not re.fullmatch(r'\d{5}', z) for z in zip_codes):
            raise ValueError('ZIP codes must be five-digit strings.')
        if month is not None and (not isinstance(month, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', month)):
            raise ValueError('Month must be YYYY-MM-DD.')
        if month is not None:
            date.fromisoformat(month)
        metadata = self.metadata()
        selected_month = month or metadata['latest_month']
        placeholders = ','.join('?' for _ in zip_codes)
        with closing(self.connect()) as conn:
            rows = [dict(r) for r in conn.execute(f"""SELECT zip_code,observed_on,zhvi,change_1y_pct,
                change_3y_pct,comparison_score,cluster,method_version FROM market_observations
                WHERE observed_on=? AND zip_code IN ({placeholders}) ORDER BY zip_code""", [selected_month, *zip_codes])]
        for row in rows:
            row['citation_id'] = f"Z{row['zip_code']}-{row['observed_on']}"
        return {'month': selected_month, 'records': rows,
                'missing_zip_codes': [z for z in dict.fromkeys(zip_codes) if z not in {r['zip_code'] for r in rows}],
                'source_url': metadata['source_url'], 'retrieved_at': metadata['retrieved_at'],
                'method': METHOD,
                'limitations': 'Historical, descriptive comparison only. ZIP areas and neighborhoods are not interchangeable. No forecasts, rental yields, school or safety ratings.'}
