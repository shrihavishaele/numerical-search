import os
import json
import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util

INDEX_NAME = "qfacts_index"
TOP_K_ES = 20

# Load BERT model once
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Elasticsearch — supports both local and Elastic Cloud
if "ES_CLOUD_ID" in st.secrets:
    es = Elasticsearch(
        cloud_id=st.secrets["ES_CLOUD_ID"],
        api_key=st.secrets["ES_API_KEY"],
    )
else:
    ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
    es = Elasticsearch(ES_HOST)

if not es.ping():
    raise SystemExit("Cannot connect to Elasticsearch")

def build_es_query(user_query):
    should = []
    filter_ = []

    if user_query.get("entity"):
        ent = user_query["entity"]
        should.append({"match": {"entity": {"query": ent, "boost": 5}}})
        should.append({"match": {"entity_class": {"query": ent, "boost": 4}}})
        should.append({"nested": {
            "path": "superclass_chain",
            "query": {
                "match": {"superclass_chain.superclass_chain": {"query": ent, "boost": 3}}
            }
        }})

    if user_query.get("context"):
        for word in user_query["context"]:
            should.append({"match": {"sentence": {"query": word, "boost": 2}}})

    if user_query.get("attribute"):
        should.append({"match": {"attribute": {"query": user_query["attribute"], "boost": 3}}})

    qcond = user_query.get("quantity_condition", {})
    if qcond:
        val = qcond.get("value")
        comp = qcond.get("comparison", "=")

        if val is not None:
            if comp == "=":
                filter_.append({"range": {"quantity.value": {"gte": val, "lte": val}}})
            elif comp == ">":
                filter_.append({"range": {"quantity.value": {"gt": val}}})
            elif comp == "<":
                filter_.append({"range": {"quantity.value": {"lt": val}}})
            elif comp == ">=":
                filter_.append({"range": {"quantity.value": {"gte": val}}})
            elif comp == "<=":
                filter_.append({"range": {"quantity.value": {"lte": val}}})
            elif comp == "between":
                hi = qcond.get("value_hi")
                if hi is not None:
                    filter_.append({"range": {"quantity.value": {"gte": val, "lte": hi}}})

    return {
        "query": {
            "bool": {
                "should": should,
                "filter": filter_,
                "minimum_should_match": 1
            }
        }
    }

def semantic_rerank(es_results, user_query, top_k=10):
    ctx_words = user_query.get("context", [])
    query_context = " ".join(ctx_words) if ctx_words else ""

    if not query_context:
        return [(0.0, h) for h in es_results[:top_k]]

    q_emb = bert_model.encode(query_context, convert_to_tensor=True)
    scored = []

    for hit in es_results:
        src = hit["_source"]
        attr = src.get("attribute", "") or ""
        if not attr:
            sim = 0.0
        else:
            d_emb = bert_model.encode(attr, convert_to_tensor=True)
            sim = util.cos_sim(q_emb, d_emb).item()
        scored.append((sim, hit))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def run_search(query_dict, top_k=10):
    """
    Returns (total_hits: int, results: list[dict]) where each result dict has:
      sentence, source, es_score, bert_score
    """
    es_query = build_es_query(query_dict)
    res = es.search(index=INDEX_NAME, body=es_query, size=max(top_k, TOP_K_ES))
    hits = res["hits"]["hits"]
    total = res["hits"]["total"]["value"] if isinstance(res["hits"]["total"], dict) else res["hits"]["total"]

    if not hits:
        return total, []

    ranked = semantic_rerank(hits, query_dict, top_k=top_k)

    results = []
    for bert_score, hit in ranked:
        src = hit["_source"]
        results.append({
            "sentence": src.get("sentence", ""),
            "source": src.get("source", src.get("url", "N/A")),
            "es_score": hit.get("_score", 0),
            "bert_score": round(bert_score, 4),
            "full_json": src,
        })
    return total, results
