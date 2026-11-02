"""Free public UI. Optional Claude mode is disabled unless explicitly enabled."""
import json
import os
from pathlib import Path
import streamlit as st
from ziprag.service import ROOT, live_service

st.set_page_config(page_title='ZipRAG · Austin, with sources', page_icon='↗', layout='wide')
st.markdown('''<style>
.stApp{background:#f8f6f0;color:#192e29}.block-container{max-width:1120px;padding-top:3rem}
h1{font-family:Georgia,serif!important;letter-spacing:-.045em;line-height:1.08!important;font-size:clamp(3rem,6vw,4.8rem)!important;font-weight:500!important}
h2,h3{color:#183e34!important}.eyebrow{font-size:.8rem;letter-spacing:.16em;font-weight:700;color:#3f685a}
[data-testid="stMetric"]{background:#fffdf8;border:1px solid #d9ddd1;border-radius:12px;padding:1rem}
a{color:#17674e}button:focus-visible,a:focus-visible{outline:3px solid #a75e18!important;outline-offset:3px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>''', unsafe_allow_html=True)


@st.cache_resource(ttl=300, show_spinner=False)
def service():
    return live_service()


engine, warning = service()
docmeta, scoremeta = engine.documents.metadata(), engine.scores.metadata()
st.markdown('<p class="eyebrow">ZIPRAG / AUSTIN REFERENCE EXPLORER</p>', unsafe_allow_html=True)
st.title('Ask Austin.\nCheck the source.')
st.write('Find neighborhood references and inspect the Analyzer’s dated ZIP-code scores. Every result links back to its evidence.')
st.caption('Free search and data mode: source excerpts and stored values, without an AI-generated answer or API charges.')
if warning:
    st.warning(warning)
a,b,c = st.columns(3)
a.metric('Sourced articles', docmeta['documents'])
b.metric('ZIP areas in snapshot', scoremeta['zip_count'])
c.metric('Housing observation date', scoremeta['latest_month'])
st.caption(f"{engine.origin} · Housing data retrieved {scoremeta['retrieved_at']}")
search_tab, methods_tab, mcp_tab = st.tabs(['Search & compare', 'Sources & evaluation', 'Connect MCP'])

with search_tab:
    choices = ['What is the history of Hyde Park?', 'Which community was built on a former airport?', 'Compare ZIP codes 78701 and 78704']
    for col, question in zip(st.columns(3), choices):
        if col.button(question, use_container_width=True):
            st.session_state['question'] = question
    with st.form('question-form'):
        question = st.text_input('Your Austin question or ZIP codes', key='question', max_chars=600,
                                 placeholder='Try a neighborhood, a landmark, or a five-digit ZIP code')
        submitted = st.form_submit_button('Search sources', type='primary')
    if submitted:
        try:
            st.session_state['evidence'] = engine.evidence(question)
        except ValueError as error:
            st.error(str(error))
            st.session_state.pop('evidence', None)
    evidence = st.session_state.get('evidence')
    if evidence:
        if evidence['scores']:
            scores = evidence['scores']
            st.header('ZIP-code evidence')
            if scores['records']:
                st.dataframe([{'ZIP code':r['zip_code'], 'Observed':r['observed_on'],
                               'Typical home value (USD)':r['zhvi'], '1-year change (%)':r['change_1y_pct'],
                               '3-year change (%)':r['change_3y_pct'], 'Comparison score (0–100)':r['comparison_score'],
                               'Citation':r['citation_id']} for r in scores['records']], hide_index=True, width='stretch')
                st.markdown(f"[Zillow Research source]({scores['source_url']}) · [Open the Analyzer](https://austin-real-estate-analyzer-d5mx2bhi5grprfmjggz83b.streamlit.app/)")
            if scores['missing_zip_codes']:
                st.info('No record for: ' + ', '.join(scores['missing_zip_codes']) + ' at ' + scores['month'] + '.')
            st.caption('Score: ' + scores['method'] + '. ' + scores['limitations'])
        st.header('Neighborhood reference excerpts')
        results = evidence['documents']['results']
        if not results:
            st.info('No matching article excerpt in this reference set. Try a neighborhood name or landmark.')
        for rank,r in enumerate(results, 1):
            with st.container(border=True):
                st.subheader(f"{rank:02d} · {r['title']}")
                st.write(r['text'])
                st.markdown(f"[Read the source article]({r['url']}) · [Saved revision](https://en.wikipedia.org/w/index.php?oldid={r['revision']})")
                st.caption(f"[{r['citation_id']}] · Retrieved {r['retrieved_at'][:10]} · Wikipedia, CC BY-SA 4.0; contributors in article history.")
        st.caption('Neighborhoods and ZIP codes are different geographic units. The reference collection does not establish an authoritative mapping between them.')
        st.download_button('Download this evidence (.json)', json.dumps(evidence, indent=2), 'ziprag-evidence.json', 'application/json')

    # The default public deployment never enables this provider-billed path.
    if os.environ.get('ZIPRAG_ALLOW_CLAUDE') == '1':
        with st.expander('Optional Claude answer'):
            st.caption('Requires your own configured API credentials and provider credits. Source search above stays free.')
            if st.button('Ask configured Claude model'):
                try:
                    from anthropic import Anthropic
                    from ziprag.claude import answer
                    api_key = os.environ.get('ANTHROPIC_API_KEY') or st.secrets.get('ANTHROPIC_API_KEY')
                    model = os.environ.get('CLAUDE_MODEL') or st.secrets.get('CLAUDE_MODEL')
                    if not api_key or not model:
                        raise ValueError('Configure a Claude API key and model ID in secure server settings.')
                    result = answer(question, engine, Anthropic(api_key=api_key, max_retries=0, timeout=25), model)
                    st.write(result['answer'])
                    with st.expander('Retrieved sources and tool calls'):
                        st.json(result)
                except ValueError as error:
                    st.error(str(error))
                except Exception:
                    st.error('Claude is unavailable. No generated answer is being substituted; use the source evidence above.')

with methods_tab:
    st.header('Evidence you can inspect')
    st.write('The free search combines SQLite FTS5 text matching with local TF-IDF vectors and cosine similarity. These are lexical vectors, not neural embeddings. Results combine both rankings and show at most one excerpt per source article.')
    st.write('The corpus contains 18 Wikipedia articles with source URLs, revision IDs, download dates, and SHA-256 checksums. Documents are chunked into overlapping passages; the build refuses altered files that do not match the manifest.')
    st.write('Housing values and descriptive scores come from the Analyzer’s read-only SQLite snapshot. Downloads from its public CloudFront route are bounded and checked with SHA-256 and SQLite integrity validation. A live-source failure is labeled before using the included dated snapshot.')
    report_path = ROOT / 'evaluation/results.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.subheader(f"Source retrieval: {report['hits_at_3']}/{report['questions']} in the top three")
        st.caption(report['limitations'])
        st.dataframe([{'Question':r['question'], 'Expected source':r['expected_source'], 'Hit@3':r['hit']} for r in report['cases']], hide_index=True, width='stretch')
    st.markdown('[Corpus license: CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [Zillow Research](https://www.zillow.com/research/data/)')
    st.caption('This demo does not establish safety, school quality, rental yields, future prices, or investment-return accuracy.')

with mcp_tab:
    st.header('The same two tools, in your MCP client')
    st.write('Run the included MCP server locally to expose document search and ZIP-score lookup to Claude Desktop or another compatible client. These read-only tools do not require model API keys. Your MCP client may have its own subscription or usage limits.')
    st.code('python mcp_server.py', language='bash')
    st.code(json.dumps({'mcpServers':{'ziprag':{'command':'/absolute/path/ZipRAG/.venv/bin/python',
            'args':['/absolute/path/ZipRAG/mcp_server.py']}}}, indent=2), language='json')
    st.caption('Replace both paths with your clone’s actual paths. This is a local stdio server, not a public remote MCP endpoint.')
    st.write('Tools: `search_documents(query, limit=3)` and `lookup_zip_scores(zip_codes, month=None)`. Provenance is available at `ziprag://provenance`.')
