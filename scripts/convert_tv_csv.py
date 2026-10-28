#!/usr/bin/env python3
"""
convert_tv_csv.py
Usage:
  python3 convert_tv_csv.py "/path/to/TradingView export.csv"
Writes a tv_YYYY-MM-DD.txt file into ~/swing-trading-bot/Data with one ticker per line (UPPERCASE).
"""
import csv, sys, os, datetime

if len(sys.argv) < 2:
    print("Usage: convert_tv_csv.py /path/to/file.csv")
    sys.exit(2)

src = os.path.expanduser(sys.argv[1])
out = os.path.expanduser(f"~/swing-trading-bot/Data/tv_{datetime.date.today().isoformat()}.txt")

# common header candidates for ticker column
candidates = ['symbol','ticker','ticker symbol','ticker_symbol','name']

symbols = []
with open(src, newline='', encoding='utf-8') as f:
    # try as CSV with header
    try:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            # find a header that looks like a ticker
            col = None
            for h in reader.fieldnames:
                lh = h.strip().lower()
                if any(c in lh for c in candidates) or lh in candidates:
                    col = h
                    break
            if col is None:
                col = reader.fieldnames[0]
            for row in reader:
                if col in row and row[col]:
                    s = row[col].strip().split()[0].strip().upper()
                    s = s.replace(':US','').replace('.US','').split('.')[0]
                    if 1 <= len(s) <= 6:
                        symbols.append(s)
        else:
            # no header: fallback to line-by-line
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = line.split(',')[0].strip().upper()
                s = s.replace(':US','').replace('.US','').split('.')[0]
                if 1 <= len(s) <= 6:
                    symbols.append(s)
    except Exception:
        # fallback: treat as simple lines
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = line.split(',')[0].strip().upper()
            s = s.replace(':US','').replace('.US','').split('.')[0]
            if 1 <= len(s) <= 6:
                symbols.append(s)

symbols = sorted(set(symbols))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as w:
    for s in symbols:
        w.write(s + "\n")

print(f"Wrote {len(symbols)} tickers to {out}")
