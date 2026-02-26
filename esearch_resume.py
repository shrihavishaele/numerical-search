import json
import os
import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers

# CONFIG
ES_HOST = "http://localhost:9200"
INDEX_NAME = "qfacts_index"
ENRICHED_PATH = r"D:\num search\fwdcodefiles\with_attributes_enriched_full.json"
BULK_CHUNK_SIZE = 500

es = Elasticsearch(ES_HOST, request_timeout=60)

if not es.ping():
    raise Exception("Cannot connect to Elasticsearch")

print("Connected to Elasticsearch")

def wait_for_index_ready(es, index, retries=10, delay=10):
    """Wait until the index is ready and queryable."""
    for attempt in range(1, retries + 1):
        try:
            es.cluster.health(index=index, wait_for_status="yellow", timeout="30s")
            return es.count(index=index)['count']
        except Exception as e:
            print(f"  Attempt {attempt}/{retries}: index not ready ({e}). Retrying in {delay}s...")
            time.sleep(delay)
    raise Exception(f"Index '{index}' did not become ready after {retries} attempts.")

# Check current document count
if es.indices.exists(index=INDEX_NAME):
    current_count = wait_for_index_ready(es, INDEX_NAME)
    print(f"Index exists with {current_count} documents. Continuing indexing...")
else:
    print("Index doesn't exist. Run esearch.py first to create the index.")
    exit(1)

# Generator that adds document ID
def generate_docs():
    line_num = 0
    with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            if line.strip():
                try:
                    doc = json.loads(line)
                    chain = doc.get("superclass_chain", [])
                    if isinstance(chain, list):
                        chain = [{"superclass_chain": c} for c in chain]
                    doc["superclass_chain"] = chain
                    
                    # Use line number as ID to avoid duplicates
                    yield {
                        "_index": INDEX_NAME,
                        "_id": line_num,  # Unique ID prevents duplicates
                        "_source": doc
                    }
                except json.JSONDecodeError:
                    continue

print("\nStarting/resuming bulk indexing...\n")

success = failed = 0
errors_log = []

for ok, info in helpers.streaming_bulk(
    es, 
    generate_docs(), 
    chunk_size=BULK_CHUNK_SIZE,
    raise_on_error=False,
    raise_on_exception=False
):
    if ok:
        success += 1
        if success % 100000 == 0:
            print(f"Indexed {success} records...")
    else:
        failed += 1
        errors_log.append(info)

print(f"\nFinished. SUCCESS: {success} | FAILED: {failed}")

if errors_log:
    error_file = r"D:\num search\fwdcodefiles\indexing_errors.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(errors_log, f, indent=2)
    print(f"Errors logged to: {error_file}")

# Check final count
final_count = wait_for_index_ready(es, INDEX_NAME)
print(f"Total documents in index: {final_count}")
print("Indexing complete!")
