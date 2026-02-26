import json
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import os

# CONFIG
ES_HOST = "http://localhost:9200"
INDEX_NAME = "qfacts_index"
ENRICHED_PATH = r"D:\num search\fwdcodefiles\merged_dataset_dedup.json"
BULK_CHUNK_SIZE = 500
CHECKPOINT_FILE = r"D:\num search\fwdcodefiles\indexing_checkpoint.txt"

# Connect to ElasticSearch
es = Elasticsearch(ES_HOST)

if not es.ping():
    raise Exception("Cannot connect to Elasticsearch")

print("Connected to Elasticsearch")

# Create index with mapping
mapping = {
    "mappings": {
        "properties": {
            "sentence": {"type": "text"},
            "entity": {"type": "text"},
            "entity_class": {"type": "text"},
            "attribute": {"type": "text"},
            "superclass_chain": {
                "type": "nested",
                "properties": {
                    "superclass_chain": {"type": "text"}
                }
            },
            "quantity": {
                "properties": {
                    "unit": {"type": "keyword"},
                    "comparison": {"type": "keyword"}
                }
            }
        }
    }
}

if es.indices.exists(index=INDEX_NAME):
    print(f"Index {INDEX_NAME} already exists, deleting it...")
    es.indices.delete(index=INDEX_NAME)

es.indices.create(index=INDEX_NAME, body=mapping)
print("Index created successfully")

# Generator function for streaming bulk
def generate_docs(skip_lines=0):
    line_num = 0
    with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            # Skip already processed lines
            if line_num <= skip_lines:
                continue
                
            if line.strip():
                try:
                    doc = json.loads(line)
                    chain = doc.get("superclass_chain", [])
                    if isinstance(chain, list):
                        chain = [{"superclass_chain": c} for c in chain]
                    doc["superclass_chain"] = chain
                    
                    yield {
                        "_index": INDEX_NAME,
                        "_source": doc
                    }
                except json.JSONDecodeError:
                    continue

# Load checkpoint if exists
skip_lines = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        skip_lines = int(f.read().strip())
    print(f"Resuming from line {skip_lines}...")

# Bulk indexing
print("\nStarting bulk indexing...\n")

success = failed = 0
errors_log = []
total_processed = skip_lines

for ok, info in helpers.streaming_bulk(
    es, 
    generate_docs(skip_lines=skip_lines), 
    chunk_size=BULK_CHUNK_SIZE,
    raise_on_error=False,  # Don't raise exception, continue processing
    raise_on_exception=False
):
    total_processed += 1
    
    if ok:
        success += 1
        if success % 100000 == 0:
            print(f"Indexed {success} records...")
            # Save checkpoint every 100k records
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(total_processed))
    else:
        failed += 1
        errors_log.append(info)
        if failed % 100 == 0:
            print(f"Failed: {failed} documents")

print(f"\nFinished indexing. SUCCESS: {success} | FAILED: {failed}")

# Save errors to file if any
if errors_log:
    error_file = r"D:\num search\fwdcodefiles\indexing_errors.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(errors_log, f, indent=2)
    print(f"Errors logged to: {error_file}")

# Clear checkpoint on successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint cleared")

print("Indexing complete!")