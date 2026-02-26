import streamlit as st
from search import run_search

st.set_page_config(page_title="QFacts Search", layout="wide")
st.title("🔍 QFacts Structured Search")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    entity = st.text_input("Entity", placeholder="e.g. trail, country")
    context_raw = st.text_input("Context (comma-separated keywords)", placeholder="e.g. length, distance")

with col2:
    comparison = st.selectbox("Comparison", ["=", ">", "<", ">=", "<=", "between"])
    value = st.number_input("Value", value=0.0, format="%g")
    value_hi = None
    if comparison == "between":
        value_hi = st.number_input("Value High", value=0.0, format="%g")
    unit = st.text_input("Unit", placeholder="e.g. km, articles")

# --- Build query ---
if st.button("Search"):
    query = {}

    if entity.strip():
        query["entity"] = entity.strip()
    if context_raw.strip():
        query["context"] = [w.strip() for w in context_raw.split(",") if w.strip()]

    qcond = {}
    if value != 0 or comparison != "=":
        qcond["value"] = value
        qcond["comparison"] = comparison
        if unit.strip():
            qcond["unit"] = unit.strip()
        if comparison == "between" and value_hi is not None:
            qcond["value_hi"] = value_hi
    if qcond:
        query["quantity_condition"] = qcond

    if not query:
        st.warning("Please fill in at least one field.")
    else:
        with st.expander("Generated Query", expanded=False):
            st.json(query)

        with st.spinner("Searching..."):
            total_hits, results = run_search(query, top_k=10)

        st.subheader(f"Total Hits: {total_hits}")

        if not results:
            st.info("No results found.")
        else:
            for i, r in enumerate(results, 1):
                with st.container():
                    st.markdown(f"**Result {i}**")
                    st.markdown(f"**Sentence:** {r['sentence']}")
                    st.markdown(f"**Source:** {r['source']}")
                    c1, c2 = st.columns(2)
                    c1.metric("ES Score", r["es_score"])
                    c2.metric("BERT Score", r["bert_score"])
                    with st.expander("View Full JSON"):
                        st.json(r["full_json"])
                    st.divider()
