from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import sys

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")


for doc in sys.stdin:
    try:
        t = json.loads(doc)
    except:
        continue
    text, i, reg = t["text"], t["id"], t["register"]
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    result = {
        "id": i,
        "text": text,
        "register": reg,
        "score": score,
        "int_score": int(round(max(0, min(score, 5)))),
    }

    print(result)
