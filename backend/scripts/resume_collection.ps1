# resume_collection.ps1 — restart every paused collection/scoring job.
# Run from backend\:  powershell -ExecutionPolicy Bypass -File scripts\resume_collection.ps1
# Each job is checkpointed and resumes where it stopped; safe to re-run anytime.

$py = ".\venv\Scripts\python.exe"

Write-Host "=== aiPHeed collection resume ===" -ForegroundColor Cyan

# 1. Scoring marathon (resumes from xlmr_scores.parquet cache)
Start-Process -NoNewWindow $py -ArgumentList "run_expanded_corpus_collection.py --sources gdelt_bq rss --resume" `
    -RedirectStandardOutput "logs_scoring.txt" -RedirectStandardError "logs_scoring_err.txt"
Write-Host "[1] Scoring marathon resumed  -> logs_scoring.txt"

# 2. Google News 2025 (2020-2024 already checkpointed; fetches only 2025)
Start-Process -NoNewWindow $py -ArgumentList "scripts/gnews_fetch_only.py --start 2020 --end 2025 --profile low" `
    -RedirectStandardOutput "logs_gnews.txt" -RedirectStandardError "logs_gnews_err.txt"
Write-Host "[2] Google News fetch resumed -> logs_gnews.txt"

# 3. GDELT REST refetch (now checkpoints per year)
Start-Process -NoNewWindow $py -ArgumentList "scripts/gdelt_fetch_only.py --start 2020 --end 2025 --window-days 30" `
    -RedirectStandardOutput "logs_gdelt.txt" -RedirectStandardError "logs_gdelt_err.txt"
Write-Host "[3] GDELT REST fetch resumed  -> logs_gdelt.txt"

# 4. National-pool enrichment (resumes from enrich_national_progress.parquet)
Start-Process -NoNewWindow $py -ArgumentList "scripts/enrich_bigquery_articles.py --src data/raw/gdelt_bq_national_only.parquet --out data/raw/gdelt_bq_national_enriched.parquet --ckpt data/raw/checkpoints_enrich/enrich_national_progress.parquet" `
    -RedirectStandardOutput "logs_enrich.txt" -RedirectStandardError "logs_enrich_err.txt"
Write-Host "[4] National enrichment resumed -> logs_enrich.txt"

Write-Host ""
Write-Host "All four jobs running. Progress:  Get-Content logs_scoring.txt -Tail 3" -ForegroundColor Green
