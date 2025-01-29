import json
import sys
import random
import io
import re

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf8', errors='replace')  # 'replace' handles non-ASCII characters

# PIPED INPUT
for line in sys.stdin:
    try:
        j = json.loads(line)
        if j["id"] != "d469177d5e577c8df9aa9a2e2bfd70fe":  # remove this from IN
            if len(re.split(r'[\s]+', j["text"])) < 500000:
                print(json.dumps(j))  #tested to be faster than .get etc
    except:
        continue
