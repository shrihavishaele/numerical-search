import json
import requests
import re
import os
import spacy
from sentence_transformers import SentenceTransformer, util

from tri import (
    wikipedia_search,
    wikipedia_title_to_qid,
    wikidata_p31,
    wikidata_occupation
)

# CONFIG
DATA_PATH = r"C:\Users\havis\Downloads\fwdcodefiles\with_attributes.json"
OUTPUT_PATH = r"C:\Users\havis\Downloads\fwdcodefiles\with_enriched_attributes.json"
CACHE_FILE = "entity_class_cache.json"

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load dataset
def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        return json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in txt.splitlines()]

records = load_json_array(DATA_PATH)
print(f"Loaded {len(records)} sentences")

# Load cache
if os.path.exists(CACHE_FILE):
    entity_class_cache = json.load(open(CACHE_FILE))
else:
    entity_class_cache = {}

def save_cache():
    json.dump(entity_class_cache, open(CACHE_FILE, "w"), indent=2)

# Main Entity Classification Logic
def get_entity_class(sentence, provided_entity):
    # Extract entities using spaCy
    doc = nlp(sentence)
    extracted_ents = [ent.text for ent in doc.ents]

    if not extracted_ents:
        return "Other", ["Other"]

    # Embed the provided entity
    prov_emb = embed_model.encode(provided_entity, convert_to_tensor=True)

    # Find spaCy entity most similar to provided entity
    best_ent = None
    best_score = -1

    for e in extracted_ents:
        e_emb = embed_model.encode(e, convert_to_tensor=True)
        score = util.cos_sim(prov_emb, e_emb)[0][0].item()
        if score > best_score:
            best_score = score
            best_ent = e

    # If no good match found: fallback
    if not best_ent:
        best_ent = extracted_ents[0]

    # Wikipedia search
    title = wikipedia_search(best_ent)
    if not title:
        return "Other", ["Other"]

    # QID
    qid = wikipedia_title_to_qid(title)
    if not qid:
        return "Other", ["Other"]

    # P31 class
    p31 = wikidata_p31(qid)

    # Human: occupation
    if p31 == ["human"]:
        occ = wikidata_occupation(qid)
        if occ:
            return occ[0], occ
        else:
            return "human", ["human"]

    # Non-human best P31
    if p31:
        main = max(p31, key=len)
        return main, p31

    return "Other", ["Other"]

# Process and save output
'''
print(get_entity_class("Microsoft"))
print(get_entity_class("Manoora"))
print(get_entity_class("Welzheim"))
print(get_entity_class("Ostrogothic"))
print(get_entity_class("John William"))
print(get_entity_class("Murray"))
print(get_entity_class("Cairns"))
'''

print("\n Predicting entity classes and writing to file...\n")

# Remove or truncate the output file before starting
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("[\n")

skip_count = 0
resume_from = 0  # Changed from 79700 to 0

for i, doc in enumerate(records):

    if i < resume_from:
        skip_count += 1
        continue

    sentence = doc.get("sentence") or doc.get("text") or ""
    provided_entity = doc.get("entity") or ""

    ent_class, superclass_chain = get_entity_class(sentence, provided_entity)
    doc["entity_class"] = ent_class
    doc["superclass_chain"] = superclass_chain

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        if i != len(records) - 1:
            f.write(",\n")

    if i % 50 == 0:
        print(f"Processed: {i}/{len(records)}")

with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    f.write("\n]")

print("\n DONE. All sentences enriched and saved!")