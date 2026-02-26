import requests
import spacy

nlp = spacy.load("en_core_web_sm")

HEADERS = {
    "User-Agent": "EntityClassifier/1.0 (https://example.com/contact)",
    "Accept": "application/json"
}


# -------------------------------------------------------------------
# SAFE JSON FETCH
# -------------------------------------------------------------------
def safe_json(url, params=None, timeout=8, headers=None):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers or HEADERS)
        return r.json()
    except:
        return None



# -------------------------------------------------------------------
# WIKIPEDIA SEARCH using list=search  (THE CORRECT ONE)
# -------------------------------------------------------------------
def wikipedia_search(query):
    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "WikidataEntityClassifier/1.0 (contact: youremail@example.com)"
    }

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "origin": "*"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        data = r.json()
    except:
        return None

    try:
        results = data["query"]["search"]
        if not results:
            return None
        return results[0]["title"]
    except:
        return None



# -------------------------------------------------------------------
# GET WIKIDATA QID for a Wikipedia page title
# -------------------------------------------------------------------
def wikipedia_title_to_qid(title):
    if not title:
        return None

    url = "https://en.wikipedia.org/w/api.php"

    data = safe_json(url, {
        "action": "query",
        "prop": "pageprops",
        "titles": title,
        "redirects": 1,
        "format": "json",
        "utf8": 1,
        "origin": "*"
    }, headers=HEADERS)

    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    return page.get("pageprops", {}).get("wikibase_item")



# -------------------------------------------------------------------
# GET P31 CLASSES FROM WIKIDATA
# -------------------------------------------------------------------
def wikidata_p31(qid):
    if not qid:
        return None

    data = safe_json(
        f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
        headers=HEADERS
    )
    if not data:
        return None

    ent = data["entities"].get(qid, {})
    claims = ent.get("claims", {})

    if "P31" not in claims:
        return None

    labels = []
    for claim in claims["P31"]:
        try:
            class_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
            class_data = safe_json(
                f"https://www.wikidata.org/wiki/Special:EntityData/{class_qid}.json",
                headers=HEADERS
            )
            labels.append(class_data["entities"][class_qid]["labels"]["en"]["value"])
        except:
            continue

    return labels if labels else None

def wikidata_occupation(qid):
    """Return P106 occupation labels for a Wikidata entity."""
    if not qid:
        return None

    data = safe_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    if not data:
        return None

    ent = data["entities"][qid]
    claims = ent.get("claims", {})

    if "P106" not in claims:
        return None

    occupations = []

    for claim in claims["P106"]:
        try:
            occ_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
            occ_data = safe_json(
                f"https://www.wikidata.org/wiki/Special:EntityData/{occ_qid}.json"
            )
            label = occ_data["entities"][occ_qid]["labels"]["en"]["value"]
            occupations.append(label)
        except:
            continue

    return occupations




# -------------------------------------------------------------------
# NER → WIKIDATA CLASSIFIER
# -------------------------------------------------------------------
def classify_sentence(sentence):
    doc = nlp(sentence)
    ents = [ent.text for ent in doc.ents]

    results = {}
    for e in ents:
        print(f"\n🔍 Looking up entity: {e}")
        title = wikipedia_search(e)
        print("   Wikipedia title:", title)

        qid = wikipedia_title_to_qid(title) if title else None
        print("   QID:", qid)

        p31 = wikidata_p31(qid) if qid else None
        if p31 == ['human']:
            occ = wikidata_occupation(qid)
            results[e] = {
                "p31": p31,
                "occupation": occ
            }
        else:
            results[e] = { "p31": p31 }

        print("   P31:", p31)

        results[e] = p31

    return results



#print(classify_sentence("Dramatically , Colbert pulled out his phone to show Tyson that the spacecraft pictures actually show that the planet is \" 20 to 30 kilometers \" than previously expected ."))
#print(classify_sentence("Email Two former Daily Show correspondents reunited Wednesday on The Late Show as Stephen Colbert was joined by Last Week Tonight host John Oliver ."))
#print(classify_sentence("Paris is the capital of France."))