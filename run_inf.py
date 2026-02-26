import sys
sys.path.append("/home/b11220100-kpal/myproject")

import json
import torch
import re
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizerFast
from tags_centric import BertSeq2SeqClassifier, get_encoder_preds, encoder_inference, create_query

# =============================
# Device setup
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Similarity model + spaCy
# =============================
nlp_coref = spacy.load("en_core_web_sm")
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
SIM_THRESHOLD = 0.80

# =============================
# BIO label mapping
# =============================
Tag = ['O','B-ent','I-ent','B-attr','I-attr','B-qty','I-qty']
ids_to_labels = {v: k for v, k in enumerate(Tag)}

# =============================
# Quantity parsing
# =============================
def parse_quantity_tuple(quantity_str):
    if not quantity_str or not isinstance(quantity_str, str):
        return {}

    q = quantity_str.strip("()")
    parts = q.split(";")

    if len(parts) != 3:
        return {}

    try:
        value = float(parts[0]) if parts[0] else None
    except ValueError:
        value = None

    unit = parts[1].strip() if parts[1] else ""
    comparison = parts[2].strip() if parts[2] else ""

    return {
        "value": value,
        "unit": unit,
        "comparison": comparison
    }

# =============================
# Pronoun replacement
# =============================
def replace_pronouns(sentence, qfact_entity):
    pronouns = ["he", "she", "it", "they", "him", "her",
                "his", "their", "them"]

    tokens = sentence.split()
    new_tokens = []

    for token in tokens:
        clean = re.sub(r'[^\w]', '', token).lower()
        if clean in pronouns:
            new_tokens.append(qfact_entity)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)

# =============================
# Entity validation
# =============================
def is_valid_entity(pred_entity, qfact_entity):
    if not pred_entity:
        return False

    emb1 = sim_model.encode(pred_entity, convert_to_tensor=True)
    emb2 = sim_model.encode(qfact_entity, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    return score >= SIM_THRESHOLD

# =============================
# Inference function
# =============================
def run_inference(sentence, quantity, model, tokenizer,
                  max_len_encoder=64, max_len_decoder=32):

    model.eval()

    encoding = tokenizer(sentence.split(),
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         padding='max_length',
                         truncation=True,
                         max_length=max_len_encoder,
                         return_tensors="pt")

    encode_ids = encoding["input_ids"].squeeze(0).to(device)
    encode_mask = encoding["attention_mask"].squeeze(0).to(device)
    offset_map = encoding["offset_mapping"].squeeze(0).to(device)

    labels = torch.zeros_like(encode_ids).to(device)
    qty_pivot = torch.logical_or(labels == 5, labels == 6).int()

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    pad_id = tokenizer.pad_token_id

    decode_ids = torch.full((max_len_decoder,), pad_id).to(device)
    decode_ids[0] = sos_id
    decode_mask = torch.zeros(max_len_decoder).to(device)
    decode_mask[0] = 1

    with torch.no_grad():
        encoder_outputs, encoder_logits = model.encode(
            input_ids=encode_ids.unsqueeze(0),
            attention_mask=encode_mask.unsqueeze(0),
            qty_pivot=qty_pivot.unsqueeze(0),
            labels=labels.unsqueeze(0)
        )

        active_tags = encoder_logits.view(-1, encoder_logits.size(-1))
        flattened_tags = torch.argmax(active_tags, axis=1)

        predLabels = get_encoder_preds(offset_map.unsqueeze(0), flattened_tags)
        enc_attribute = encoder_inference(sentence, predLabels)

        token_label_ids = torch.argmax(encoder_logits, dim=-1)[0].cpu().tolist()
        offsets_list = encoding["offset_mapping"][0].cpu().tolist()
        words = sentence.split()

        word_level_labels = []
        for i, (start, end) in enumerate(offsets_list):
            if start == 0 and end != 0:
                lab_name = ids_to_labels.get(token_label_ids[i], "O")
                word_level_labels.append(lab_name)

        n = min(len(words), len(word_level_labels))
        words = words[:n]
        word_level_labels = word_level_labels[:n]

        bio_entity = " ".join(
            [w for w, t in zip(words, word_level_labels)
             if t in ("B-ent", "I-ent")]
        ).strip()

    return {
        "sentence": sentence,
        "entity": bio_entity,
        "attribute": enc_attribute
    }

# =============================
# QFact Processing with NULL fix
# =============================
def process_qfact_dataset(input_file, model, tokenizer,
                          attr_file="with_attributes_full.json"):

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(attr_file, "a", encoding="utf-8") as attr_out:

        for line in infile:
            record = json.loads(line.strip())

            qfact_entity = record.get("entity") or record.get("entityStr")
            qfacts = record.get("Qfacts", [])

            for qfact in qfacts:

                sentence = qfact["sentence"]
                quantity_str = qfact.get("quantityStr", "")
                quantity_struct = parse_quantity_tuple(qfact.get("quantity", ""))

                pred = run_inference(sentence, quantity_str, model, tokenizer)

                # =============================
                # NULL ENTITY FIX
                # =============================
                if not pred["entity"]:

                    doc_spacy = nlp_coref(sentence)
                    has_pronoun = any(token.pos_ == "PRON"
                                      for token in doc_spacy)

                    if has_pronoun and qfact_entity:

                        modified_sentence = replace_pronouns(
                            sentence,
                            qfact_entity
                        )

                        new_pred = run_inference(
                            modified_sentence,
                            quantity_str,
                            model,
                            tokenizer
                        )

                        if is_valid_entity(new_pred["entity"], qfact_entity):

                            output = {
                                "sentence": sentence,
                                "entity": qfact_entity,
                                "attribute": new_pred["attribute"],
                                "quantity": quantity_struct
                            }

                            attr_out.write(
                                json.dumps(output, ensure_ascii=False) + "\n"
                            )
                            continue

                    # Skip if not recoverable
                    continue

                # =============================
                # NORMAL CASE
                # =============================
                output = {
                    "sentence": sentence,
                    "entity": pred["entity"],
                    "attribute": pred["attribute"],
                    "quantity": quantity_struct
                }

                attr_out.write(
                    json.dumps(output, ensure_ascii=False) + "\n"
                )

# =============================
# Load model
# =============================
tokenizer = BertTokenizerFast.from_pretrained(
    "/home/b11220100-kpal/myproject/bert-base-uncased"
)
tokenizer.add_tokens(["<EOS>", "<SOS>"])

model = BertSeq2SeqClassifier(
    tokenizer=tokenizer,
    decoder_num_labels=30524,
    num_decoder_layers=4,
    freeze_bert=True,
    decoder_max_len=3
).to(device)

checkpoint = torch.load(
    "/home/b11220100-kpal/myproject/model-attribute-aware.pt",
    map_location=device
)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# =============================
# Run
# =============================
process_qfact_dataset(
    input_file="/home/b11220100-kpal/myproject/oelp_sem_6/qfact.json",
    model=model,
    tokenizer=tokenizer
)

print("DONE with NULL entity fix.")