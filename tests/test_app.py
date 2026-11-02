import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from ziprag.service import ROOT, ZipRAG


class AppTests(unittest.TestCase):
    def test_free_app_search_and_numeric_evidence(self):
        with patch('ziprag.service.live_service', return_value=(ZipRAG(),None)):
            app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value,'18')
            app.text_input(key='question').set_value('Compare ZIP codes 78701 and 78704')
            next(b for b in app.button if b.label == 'Search sources').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.dataframe)
            self.assertEqual(len(app.dataframe[0].value),2)
