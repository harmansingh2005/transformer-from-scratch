import os, io, random

RAW_FILE = os.path.join("raw", "en_es.tsv")
OUT_DIR = "data"

def read_pairs(path):
    pairs = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            _, en, _, es = parts[:4]
            en, es = en.strip(), es.strip()
            if en and es:
                pairs.append((en, es))
    return pairs

def dedupe_pairs(pairs, one_per_en=True):
    if one_per_en:
        seen = set()
        out = []
        for en, es in pairs:
            if en not in seen:
                out.append((en, es))
                seen.add(en)
        return out
    else:
        return list(set(pairs))

def filter_by_len(pairs, max_len=128):
    out = []
    for en, es in pairs:
        if len(en.split()) <= max_len and len(es.split()) <= max_len:
            out.append((en, es))
    return out

def write_parallel(pairs, out_dir, split=0.1, seed=42):
    random.Random(seed).shuffle(pairs)
    cut = max(1, int(len(pairs) * (1 - split)))
    train, valid = pairs[:cut], pairs[cut:]

    os.makedirs(out_dir, exist_ok=True)

    def dump(fname, lines):
        with io.open(os.path.join(out_dir, fname), "w", encoding="utf-8") as w:
            for s in lines:
                w.write(s + "\n")

    dump("train.src", [en for en, _ in train])
    dump("train.tgt", [es for _, es in train])
    dump("valid.src", [en for en, _ in valid])
    dump("valid.tgt", [es for _, es in valid])

    print(f"Wrote {len(train)} train pairs and {len(valid)} valid pairs into '{out_dir}/'")

def main():
    if not os.path.exists(RAW_FILE):
        raise SystemExit(f"❌ Could not find raw file: {RAW_FILE}")
    pairs = read_pairs(RAW_FILE)
    pairs = dedupe_pairs(pairs, one_per_en=True)
    pairs = filter_by_len(pairs, max_len=128)
    if not pairs:
        raise SystemExit("❌ No usable sentence pairs found!")
    write_parallel(pairs, OUT_DIR, split=0.1)

if __name__ == "__main__":
    main()
