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
        print(json.dumps(j))  #tested to be faster than .get etc
    except:
        continue
