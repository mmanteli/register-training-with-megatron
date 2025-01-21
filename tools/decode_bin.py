#from Sampo https://gist.github.com/spyysalo/efbc21628da4a7ad02d1f90ed310851e
#  module use /appl/local/csc/modulefiles
# module load pytorch/2.4


import sys
import struct

import numpy as np

from argparse import ArgumentParser

from transformers import AutoTokenizer


_INDEX_HEADER = b"MMIDIDX\x00\x00"


DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}

import math

millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def argparser():
    ap = ArgumentParser()
    ap.add_argument('path', help='data path without suffix (.idx/.bin)')
    ap.add_argument('tokenizer')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--stop', type=int, default=None)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    #print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    idxfn, binfn = f'{args.path}.idx', f'{args.path}.bin'

    # Load index
    with open(idxfn, 'rb') as f:
        header = f.read(len(_INDEX_HEADER))
        assert header == _INDEX_HEADER
        version = struct.unpack('<Q', f.read(8))
        assert version == (1,)

        code = struct.unpack('<B', f.read(1))[0]
        dtype = DTYPE_MAP[code]

        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]

        sequence_lengths = np.fromfile(f, np.int32, sequence_count)
        sequence_pointers = np.fromfile(f, np.int64, sequence_count)
        document_indices = np.fromfile(f, np.int64, document_count)
        total_tokens=np.sum(sequence_lengths)

        print(f'''read .idx {idxfn}:
dtype         : {dtype.__name__}
sequence_count: {sequence_count}
document_count: {document_count}
sequence_lengths (shape {sequence_lengths.shape}): {sequence_lengths}
total_tokens: {millify(total_tokens)}, ({total_tokens})
sequence_pointers (shape {sequence_pointers.shape}): {sequence_pointers}
document_indices (shape {document_indices.shape}): {document_indices}''', file=sys.stderr)

    
    # Load and print decoded .bin records
    #with open(binfn, 'rb') as f:
    #    stop = args.stop if args.stop is not None else sequence_count
    #    for i in range(args.start, stop, args.step):
    #        offset, length = sequence_pointers[i], sequence_lengths[i]
    #        f.seek(offset)
    #        data = np.fromfile(f, dtype, length)
    #        print('-'* 30, i, '-'*30)
    #        print(tokenizer.decode(data))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
