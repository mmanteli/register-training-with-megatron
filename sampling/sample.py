import json
import sys
import random
import io
import os
from argparse import ArgumentParser
#from filter_by_register import assign_labels, is_hybrid, argparser


# Registers used in the experiment.
# note: if you put sublabels here, the end of the loop, just before saving, requires special cases
registers = ["dtp","HI","HI-IN","ID","IN","IP","MT","NA","ne","OP","SP","LY", "no-label"]


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


# factor of data to take; there is too much "IN" for example in the data, so we do not want all of it
# approximated from HPLT statistics, to 150B so that we get at least 100B
limits_en = {
	"dtp": 0.18732782369146006,
	"HI": 1,
	"ID": 0.47222222222222222,
	"IN": 0.18428184281842822,
	"IP": 0.2833333333333333,
	"MT": 0.5,
	"NA": 0.29629629629629634,
	"ne": 0.2504604051565377,
	"OP": 0.3215130023640662,
	"SP": 1,
	"LY": 1,
	"no-label": 0.8,
	"HI-IN": 1,
}
limits = {"eng_Latn": limits_en}

# this for testing
label_counts={k:0 for k in registers}

def argparser():
    ap = ArgumentParser()
    ap.add_argument("--registers", default=registers, help="R1[,R2...]")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--exclude_hybrids", default=True, action="store_true")
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

# PIPED INPUT
for line in sys.stdin:   # this is a double line
        texts,labels = line.split(args.sep)
        textd = json.loads(texts)
        if len(textd["text"]) < args.length_limit:
            continue
        labeld = json.loads(labels)
        assert textd["id"] == labeld["id"], "id mismatch"

        probabilities = labeld["register_probabilities"]
        r = assign_labels(probabilities, args.threshold)
        if len(r) == 0:   # no label
            register="no-label"
        elif args.exclude_hybrids and is_hybrid(r):
            if r == set(["HI","IN"]):
                register = "HI-IN"
            else:
                continue
        else: 
            register = '-'.join([j for j in r if j in args.registers])  # remove sublabel here, but save it in file
            # for the two exceptions
            if register=="NA-ne" or register=="ne-NA":
                register="ne"
            if register=="IN-dtp" or register=="dtp-IN":
                register="dtp"

        # because we do not want the entire 23T, we saple down with register specific limits
        if random.random() < limits[args.lang][register]:
            print(json.dumps({"id":textd["id"], "text":textd["text"], "register":[*r]}, ensure_ascii=False), file=filenames[register])  #tested to be faster than .get etc
            # hpltver1.0 does not have id, so make it manually or save url or somthing then
            #label_counts[register]+=1

for k,v in filenames.items():
    v.close()
#print(label_counts)
