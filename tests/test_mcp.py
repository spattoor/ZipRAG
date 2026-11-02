import json
import sys
import unittest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ziprag.service import ROOT


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_handshake_and_both_tools(self):
        params = StdioServerParameters(command=sys.executable,args=[str(ROOT/'mcp_server.py')])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as client:
                await client.initialize()
                tools = await client.list_tools()
                self.assertEqual({t.name for t in tools.tools},{'search_documents','lookup_zip_scores'})
                documents = await client.call_tool('search_documents', {'query':'Which neighborhood contains Ramsey Park?'})
                self.assertFalse(documents.isError)
                text = json.loads(documents.content[0].text)
                self.assertIn('Rosedale',text['results'][0]['title'])
                scores = await client.call_tool('lookup_zip_scores', {'zip_codes':['78701']})
                self.assertFalse(scores.isError)
                data = json.loads(scores.content[0].text)
                self.assertEqual(data['records'][0]['zip_code'],'78701')
                provenance = await client.read_resource('ziprag://provenance')
                self.assertIn('documents', json.loads(provenance.contents[0].text))
