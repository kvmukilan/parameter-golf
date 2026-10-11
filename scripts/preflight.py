#!/usr/bin/env python3
"""Validate training inputs before renting GPUs; no ML dependencies or downloads."""
import argparse
import json
from pathlib import Path
import struct
import sys


def check_shard(path):
    with path.open('rb') as source:
        header = source.read(1024)
    if len(header) != 1024:
        raise ValueError(f'{path}: incomplete shard header')
    magic, version, tokens = struct.unpack_from('<iii', header)
    if magic != 20240520 or version != 1 or tokens <= 0:
        raise ValueError(f'{path}: invalid shard header')
    if path.stat().st_size != 1024 + tokens * 2:
        raise ValueError(f'{path}: truncated or oversized token data')
    return tokens


def check_inputs(data, tokenizer):
    if not tokenizer.is_file() or tokenizer.stat().st_size == 0:
        raise ValueError(f'{tokenizer}: missing or empty tokenizer')
    result = {}
    for split in ('train', 'val'):
        shards = sorted(data.glob(f'fineweb_{split}_*.bin'))
        if not shards:
            raise ValueError(f'{data}: no {split} shards')
        result[split] = {'shards': len(shards), 'tokens': sum(check_shard(p) for p in shards)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/datasets/fineweb10B_sp1024'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tokenizers/fineweb_1024_bpe.model'))
    args = parser.parse_args()
    try:
        print(json.dumps(check_inputs(args.data, args.tokenizer), indent=2))
    except (OSError, ValueError) as error:
        print(f'Preflight failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
