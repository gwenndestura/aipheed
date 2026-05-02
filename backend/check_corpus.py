# save as check_data.py
import pandas as pd
from pathlib import Path

checkpoint_dir = Path("data/raw/checkpoints")

print("="*60)
print("COLLECTED DATA (Checkpoints)")
print("="*60)

total = 0
for f in checkpoint_dir.glob("*.parquet"):
    df = pd.read_parquet(f)
    count = len(df)
    total += count
    print(f"\n📁 {f.name}: {count} articles")
    
    # Show source type
    if 'fetcher_source' in df.columns:
        sources = df['fetcher_source'].unique()
        print(f"   Source: {', '.join(sources)}")
    
    # For GDELT, check if it has full text (BigQuery) or not (REST)
    if 'gdelt' in f.name and 'summary' in df.columns and count > 0:
        has_text = df['summary'].str.len().fillna(0) > 100
        if has_text.sum() > 0:
            print(f"   ✅ BigQuery mode: {has_text.sum()} articles with FULL TEXT")
        else:
            print(f"   ⚠️ REST mode: No full text (titles only)")
    
    # Show a sample
    if count > 0 and 'title' in df.columns:
        print(f"   Sample: {df['title'].iloc[0][:60]}...")

print(f"\n{'='*60}")
print(f"TOTAL ARTICLES COLLECTED: {total}")
print(f"{'='*60}")