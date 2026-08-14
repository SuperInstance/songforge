#!/usr/bin/env python3
"""Vocabulary diversity (type-token ratio) for a lyrics file.
Usage: python3 vocab_diversity.py <file>
"""
import sys
import re

def ttr(path):
    text = open(path).read().lower()
    words = re.findall(r"[a-z']+", text)
    if not words:
        return 0.0, 0, 0
    types = set(words)
    return len(types) / len(words), len(types), len(words)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        r, t, w = ttr(p)
        print(f"{p}: TTR={r:.3f} (types={t}, tokens={w})")
