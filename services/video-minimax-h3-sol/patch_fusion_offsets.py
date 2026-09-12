"""Promote Triton element offsets before multiplication; preserve pinned source."""
import hashlib
from pathlib import Path
import sys

BEFORE = '28f47ad80871d9873c4a3b3ad6f89e75a482088375a97d408ebaf2857fab391b'
AFTER = 'af63683f9554076979b8d44adcffcca46d4c58fbe317d7144a7b4aafe2443073'

def apply(path):
    path = Path(path)
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == AFTER:
        return
    if digest != BEFORE:
        raise ValueError('unexpected Sol-H3 fusion source revision')
    source = original.decode()
    for name in ('row', 'pid'):
        source = source.replace(f'{name} = tl.program_id(0)', f'{name} = tl.program_id(0).to(tl.int64)')
    patched = source.encode()
    if hashlib.sha256(patched).hexdigest() != AFTER:
        raise ValueError('unexpected fusion patch output')
    path.write_bytes(patched)

if __name__ == '__main__':
    apply(sys.argv[1])
