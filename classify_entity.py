import json
import os
import time
from tri import (
    wikipedia_search,
    wikipedia_title_to_qid,
    wikidata_p31,
    wikidata_occupation
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

INPUT_FILE = "entities_100k_200k.txt"
OUTPUT_FILE = "entity_classes_100k_200k.json" 
CACHE_FILE = "entity_class_cache_100k_200k.json"

SLEEP_TIME = 0.01   # delay between API calls (avoid rate limits)
MAX_WORKERS = 10    # number of parallel workers
CACHE_SAVE_INTERVAL = 500  # save cache every N entities

# Load cache if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        entity_cache = json.load(f)
else:
    entity_cache = {}

# Add thread-safe lock for cache
cache_lock = Lock()

def save_cache():
    with cache_lock:
        with open(CACHE_FILE, "w") as f:
            json.dump(entity_cache, f, indent=2)


def get_entity_class(entity):

    # Use cache (thread-safe check)
    with cache_lock:
        if entity in entity_cache:
            return entity_cache[entity]

    try:
        # Wikipedia search
        title = wikipedia_search(entity)
        if not title:
            result = ("Other", ["Other"])
            with cache_lock:
                entity_cache[entity] = result
            return result

        # QID
        qid = wikipedia_title_to_qid(title)
        if not qid:
            result = ("Other", ["Other"])
            with cache_lock:
                entity_cache[entity] = result
            return result

        # P31 class
        p31 = wikidata_p31(qid)

        # Human → occupation
        if p31 == ["human"]:
            occ = wikidata_occupation(qid)
            if occ:
                result = (occ[0], occ)
            else:
                result = ("human", ["human"])
        elif p31:
            main = max(p31, key=len)
            result = (main, p31)
        else:
            result = ("Other", ["Other"])

    except Exception:
        result = ("Other", ["Other"])

    with cache_lock:
        entity_cache[entity] = result
    
    # Sleep only for API calls
    time.sleep(SLEEP_TIME)
    return result


print("Loading entities...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entities = [line.strip() for line in f if line.strip()]

print(f"Total entities: {len(entities)}")

# Filter out already cached entities
uncached_entities = [e for e in entities if e not in entity_cache]
print(f"Already cached: {len(entities) - len(uncached_entities)}")
print(f"To process: {len(uncached_entities)}")

processed = 0

# Process with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_entity = {executor.submit(get_entity_class, entity): entity 
                       for entity in uncached_entities}
    
    # Process completed tasks
    for future in as_completed(future_to_entity):
        entity = future_to_entity[future]
        try:
            future.result()  # This will raise exception if task failed
            processed += 1
            
            # Print progress every 100
            if processed % 100 == 0:
                print(f"Processed: {processed}/{len(uncached_entities)}")
            
            # Save progress periodically
            if processed % CACHE_SAVE_INTERVAL == 0:
                save_cache()
                
        except Exception as e:
            print(f"Error processing {entity}: {e}")
            processed += 1

# Final save
save_cache()

# Save clean output file
with open(OUTPUT_FILE, "w") as out:
    json.dump(entity_cache, out, indent=2)

print("\nDONE.")
print(f"Total in cache: {len(entity_cache)}")
print(f"Newly processed: {processed}")