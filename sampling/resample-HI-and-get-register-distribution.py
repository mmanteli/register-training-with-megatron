import json
import sys
import random
import io
import os
from argparse import ArgumentParser
#from filter_by_register import assign_labels, is_hybrid, argparser


# Registers used in the experiment.
# note: if you put sublabels here, the end of the loop, just before saving, requires special cases
registers =["HI"]


# mappings between labels
LABEL_HIERARCHY = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}
LABEL_PARENT = {c: p for p, cs in LABEL_HIERARCHY.items() for c in cs}


# USING ONLY HI HERE
limits_en = {
	"HI": 1,
}
limits = {"eng_Latn": limits_en}


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--registers", default=registers, help="R1[,R2...]")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--exclude_hybrids", default=False, action="store_true")
    ap.add_argument("--length_limit", type=int, default=200, help="doc len limit in char")
    ap.add_argument("--file_prefix", default="", help="prefix for files, for splits, different dirs etc.")
    ap.add_argument("--file_suffix", default="", help="suffix for files, for sharding (add _ yourself!)")
    return ap

def is_hybrid(labels):
    if len(labels) > 2:
        return True
    if len(labels) == 2:
        l1, l2 = labels
        return not (
            l1 in LABEL_PARENT
            and LABEL_PARENT[l1] == l2
            or l2 in LABEL_PARENT
            and LABEL_PARENT[l2] == l1
        )
    return False


def assign_labels(probabilities, threshold):
    labels = set()
    for label, prob in probabilities.items():
        if prob >= threshold:
            labels.add(label)
            if label in LABEL_PARENT:
                # assure that parent also included
                labels.add(LABEL_PARENT[label])
    return labels



args = argparser().parse_args(sys.argv[1:])
args.sep = '\t'
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
for k in registers:
    os.makedirs(f'results/{args.lang}/{k}', exist_ok=True)
filenames = {k:open(f'results/{args.lang}/{k}/{args.file_prefix}{args.lang}_{k}{args.file_suffix}.jsonl', 'a') for k in registers}

collection = []
# PIPED INPUT
for line in sys.stdin:   # this is a double line
        texts,labels = line.split(args.sep)
        textd = json.loads(texts)
        #if len(textd["text"]) < args.length_limit:
        #    continue
        labeld = json.loads(labels)
        assert textd["id"] == labeld["id"], "id mismatch"

        probabilities = labeld["register_probabilities"]
        r = assign_labels(probabilities, args.threshold)
        register = "-".join([j for j in r]) 

        if register == "HI" or register == "HI-re" or register == "re-HI":
            print(json.dumps({"id":textd["id"], "text":textd["text"], "register":[*r]}, ensure_ascii=False), file=filenames["HI"])  #tested to be faster than .get etc
        collection.append(register)

for k,v in filenames.items():
    v.close()
with open(f"results/register_dist_{args.file_suffix}.csv", "w") as f:
    for line in collection:
        f.write(f"{line}\n")
#print(label_counts)
