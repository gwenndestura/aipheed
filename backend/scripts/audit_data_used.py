"""Quick audit of what data fed the trained LightGBM model."""
import pandas as pd

features = pd.read_parquet("data/processed/features_fused.parquet")
print("=== FEATURE MATRIX (primary + secondary fused) ===")
print(f"Total rows: {len(features)}")
print(f"Provinces:  {features['province_code'].nunique()} (expected 5)")
print(f"Quarters:   {features['quarter'].nunique()} (expected 24)")
print(f"First quarter: {features['quarter'].min()}")
print(f"Last quarter:  {features['quarter'].max()}")
print(f"Feature columns: {len(features.columns) - 2}")

corpus = pd.read_parquet("data/processed/corpus_geocoded.parquet")
print()
print("=== NEWS CORPUS (secondary data) ===")
print(f"Total articles collected: {len(corpus)}")
geocoded = corpus[corpus["province_code"].notna()]
print(f"Geocoded to CALABARZON:   {len(geocoded)} ({len(geocoded)/len(corpus)*100:.1f}%)")

in_window = geocoded[geocoded["quarter"].between("2020-Q1", "2025-Q4")]
print(f"In 2020-2025 window:      {len(in_window)}")
print(f"Avg per province-quarter: {len(in_window)/120:.1f}")

print()
print("=== Articles per province (2020-2025) ===")
prov_map = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}
counts = in_window["province_code"].value_counts().sort_index()
for code, n in counts.items():
    print(f"  {prov_map.get(code, code):12s}  {n:5d} articles")

print()
print("=== Articles per year (2020-2025) ===")
in_window2 = in_window.copy()
in_window2["year"] = in_window2["quarter"].str.split("-").str[0]
print(in_window2["year"].value_counts().sort_index())
