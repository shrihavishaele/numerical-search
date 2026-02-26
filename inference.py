# r"C:\Users\student\Downloads\model-tag_centric.pt"
# r"C:\Users\student\Downloads\model-tag_centric.pt"
import sys
sys.path.append("/home/b11220100-kpal/myproject/oelp_sem6") 
import json
import torch
from tags_centric import BertSeq2SeqClassifier, get_encoder_preds, encoder_inference, create_query
from transformers import BertTokenizerFast

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# BIO label mapping
Tag = ['O','B-ent','I-ent','B-attr','I-attr','B-qty','I-qty']
ids_to_labels = {v: k for v, k in enumerate(Tag)}

def parse_quantity_tuple(quantity_str):
    """
    Parse Qfact 'quantity' field (e.g., '(1.000;several inhabitants;=)')
    into a structured dict: {"value": 1.0, "unit": "...", "comparison": "="}.
    """
    if not quantity_str or not isinstance(quantity_str, str):
        return {}

    q = quantity_str.strip("()")  # remove parentheses
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

def run_inference(sentence, quantity, model, tokenizer, max_len_encoder=64, max_len_decoder=32, verbose=False):
    """
    Improved inference:
      - extracts word-level BIO labels for entity & attribute using offset mapping,
      - performs greedy autoregressive decoding using logits[:, -1, :],
      - uses tokenizer's SOS/EOS token ids (no magic numbers),
      - returns bio entity, bio attribute and generative attribute.
    """
    # Tokenize inputs (single example)
    model.eval()

    encoding = tokenizer(sentence.split(),
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         padding='max_length',
                         truncation=True,
                         max_length=max_len_encoder,
                         return_tensors="pt")

    #encode_ids = encoding["input_ids"].to(device)
    #encode_mask = encoding["attention_mask"].to(device)
    #offset_map = encoding["offset_mapping"].to(device)

    #dummy_labels = torch.zeros_like(encode_ids).to(device)
    #qty_pivot = torch.logical_or(dummy_labels == 5, dummy_labels == 6).int().to(device)

    encode_ids = encoding["input_ids"].squeeze(0).to(device, dtype=torch.long)
    encode_mask = encoding["attention_mask"].squeeze(0).to(device, dtype=torch.long)
    offset_map = encoding["offset_mapping"].squeeze(0).to(device, dtype=torch.long)

    labels = torch.zeros_like(encode_ids).to(device, dtype=torch.long)
    qty_pivot = torch.logical_or(labels == 5, labels == 6).int().to(device)

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    if sos_id is None or eos_id is None:
        raise ValueError("Tokenizer missing <SOS>/<EOS> tokens")
    #decode_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)

    pad_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<PAD>")
    decode_ids = torch.full((max_len_decoder,), pad_id, dtype=torch.long).to(device)
    decode_ids[0] = sos_id
    decode_mask = torch.zeros(max_len_decoder, dtype=torch.long).to(device)
    decode_mask[0] = 1

    batch = (
        {
            'input_ids': encode_ids.unsqueeze(0),
            'attention_mask': encode_mask.unsqueeze(0),
            'labels': labels.unsqueeze(0),
            'qty_pivot': qty_pivot.unsqueeze(0),
            'offset_mapping': offset_map.unsqueeze(0)
        },
        {
            'input_ids': decode_ids.unsqueeze(0),
            'attention_mask': decode_mask.unsqueeze(0)
        }
    )

    sentence = sentence.strip()
    encode_ids = batch[0]['input_ids'].to(device, dtype=torch.long)
    encode_mask = batch[0]['attention_mask'].to(device, dtype=torch.long)
    labels = batch[0]['labels'].to(device, dtype=torch.long)
    qty_pivot = batch[0]['qty_pivot'].to(device, dtype=torch.long)
    offset_map = batch[0]['offset_mapping'].to(device, dtype=torch.long)
    decode_ids = batch[1]['input_ids'].to(device, dtype=torch.long)
    decode_mask = batch[1]['attention_mask'].to(device, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs, encoder_logits = model.encode(
            input_ids=encode_ids,
            attention_mask=encode_mask,
            qty_pivot=qty_pivot,
            labels=labels
        )

        active_tags = encoder_logits.view(-1, encoder_logits.size(-1))
        flattened_tags = torch.argmax(active_tags, axis=1)

        predLabels = get_encoder_preds(offset_map, flattened_tags)
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
        words, word_level_labels = words[:n], word_level_labels[:n]

        bio_entity = " ".join(
             [w for w, t in zip(words, word_level_labels) if t in ("B-ent", "I-ent")]
         ).strip()


        query = create_query(encode_ids, flattened_tags, decode_ids)
        #ys = query[:, :1].clone().to(device)
        #ys = torch.ones(1, 1, device=device) * 30523
        #ys = torch.ones(1, 1, device=device) * sos_id
        #ys = (torch.ones(1, 1)*30523).to(device)
        ys = torch.full((1,1), sos_id, dtype=torch.long, device=device)
        preds = []

        for i in range(max_len_decoder):
            
            decoded_outputs = model.decode(
                input_ids=encode_ids,
                labels=flattened_tags,
                encoder_outputs=encoder_outputs,
                encoder_logits=encoder_logits,
                target_ids=ys
            )

            prob = model.generate(decoded_outputs)
            next_word_probs = torch.softmax(prob, dim=-1)
            _, next_word_index = torch.max(next_word_probs, dim=-1)

            if next_word_index[:,i]==30522:
                    break
            next_word = tokenizer.convert_ids_to_tokens(next_word_index[:,i])

            next_word_index_tensor = torch.ones(1, 1, device=device) * next_word_index[:, i]
            preds.append(next_word[0])

            # Append token id for next iteration
            ys = torch.cat([ys, next_word_index_tensor.to(device)], dim=1)
    

    gen_attr = " ".join(preds).strip()
    final_attr = f"{gen_attr} {enc_attribute}".strip()

    if verbose:
        print(f"Sentence: {sentence}")
        print(f"Quantity: {quantity}")
        print(f"Final attr: {final_attr}")

    return {
        "sentence": sentence,
        "entity": bio_entity,
        "attribute": final_attr
    }

def process_qfact_dataset(input_file, model, tokenizer,
                          attr_file="with_attributes.json",
                          null_file="null_attributes.json",
                          summary_file="summary.json",
                          limit=None,
                          skip_count = 0,
                          verbose=False):
    """
    Process Qfact dataset with tag_centric inference and split outputs.
    Dataset structure:
    {
      "entity": "<Entity>",
      "Qfacts": [
         {"sentence": ..., "quantityStr": ..., "quantity": {...}, ...},
         ...
      ]
    }
    """
    with_attr_count = 0
    null_attr_count = 0
    null_with_unit_count = 0
    processed = 0
    sentence_count = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(attr_file, "a", encoding="utf-8") as attr_out, \
         open(null_file, "a", encoding="utf-8") as null_out:

        for line in infile:
            record = json.loads(line.strip())
            qfacts = record.get("Qfacts", [])

            for qfact in qfacts:
                if processed < skip_count:
                    processed += 1
                    continue
                if limit is not None and processed >= limit:
                    break

                sentence = qfact["sentence"]
                quantity_str = qfact.get("quantityStr", "")
                quantity_struct = parse_quantity_tuple(qfact.get("quantity", ""))

                pred = run_inference(sentence, quantity_str, model, tokenizer, verbose=verbose)

                sentence_count += 1
                if sentence_count % 1000 == 0:
                    print(f"Processed {sentence_count} sentences...")

                if not pred["attribute"] or pred["attribute"] == "unknown":
                    processed += 1
                    if isinstance(quantity_struct, dict):
                        unit = quantity_struct.get("unit", "").strip()
                    if unit:
                        null_with_unit_count += 1
                    continue

                if not pred["entity"] or pred["entity"] == "unknown":
                    processed += 1
                    continue

                output = {
                    "sentence": sentence,
                    "entity": pred["entity"],
                    "attribute": pred["attribute"],
                    "quantity": quantity_struct
                }

                with_attr_count += 1
                attr_out.write(json.dumps(output, ensure_ascii=False) + "\n")

                processed += 1

            if limit is not None and processed >= limit:
                break

    # Save summary
    summary = {
        "with_attribute": with_attr_count,
        "without_attribute": null_attr_count,
        "without_attribute_but_with_unit": null_with_unit_count
    }
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    return summary


# --- Load tokenizer and model ---
tokenizer = BertTokenizerFast.from_pretrained("/home/b11220100-kpal/myproject/bert-base-uncased")
tokenizer.add_tokens(["<EOS>", "<SOS>"])

model = BertSeq2SeqClassifier(
    tokenizer=tokenizer,
    decoder_num_labels=30524,
    num_decoder_layers=4,
    freeze_bert=True,
    decoder_max_len=3
).to(device)

checkpoint = torch.load("/home/b11220100-kpal/myproject/model-attribute-aware.pt", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

skip_lines = 10000
# Run on Qfact dataset 
summary = process_qfact_dataset(
    input_file="/home/b11220100-kpal/myproject/qfact.json", 
    model=model,
    tokenizer=tokenizer,
    limit=None,
    skip_count = skip_lines
)

print("Summary:", json.dumps(summary, indent=2))
