import fasttext
import sys
import json
import ast

model = fasttext.load_model("openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")

for doc in sys.stdin:
    try:
        t = ast.literal_eval(doc)
        t = dict(t)
    except:
        continue
    text, i, reg, edu_score, edu_int_score = t["text"], t["id"], t["register"], t["score"], t["int_score"]
    text_ = text.replace("\n", " ")
    label, prob = model.predict(text_, k=1)
    result = {
        "id": i,
        "text": text,
        "register": reg,
        "score": edu_score,
        "int_score": edu_int_score,
        "dclm-label": label[0],
        "dclm-prob": prob[0],
    }
    print(json.dumps(result))
