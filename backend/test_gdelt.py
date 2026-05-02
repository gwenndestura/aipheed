"""
Test GDELT BigQuery fetcher on local Windows machine
"""
import hashlib
import json
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account

# Path to your downloaded key file
KEY_PATH = r"C:\Users\admin\Downloads\angel_vhea014\angel_vhea014\service-account-key.json"

print("🔑 Loading credentials...")
credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

print("📊 Connecting to BigQuery...")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Simple test query for CALABARZON food articles
query = """
SELECT 
    DATE(_PARTITIONTIME) as date,
    DocumentIdentifier as url,
    V2Enhanced as text,
    V2Tone as tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE 
    _PARTITIONTIME >= '2024-01-01'
    AND _PARTITIONTIME <= '2024-01-31'
    AND (
        LOWER(V2Enhanced) LIKE '%calabarzon%'
        OR LOWER(V2Enhanced) LIKE '%batangas%'
        OR LOWER(V2Enhanced) LIKE '%laguna%'
        OR LOWER(V2Enhanced) LIKE '%cavite%'
        OR LOWER(V2Enhanced) LIKE '%rizal%'
        OR LOWER(V2Enhanced) LIKE '%quezon%'
    )
    AND LOWER(V2Enhanced) LIKE '%food%'
LIMIT 5
"""

print("🚀 Running query (this may take 10-20 seconds)...")
try:
    query_job = client.query(query)
    results = list(query_job.result())
    
    print(f"\n✅ Found {len(results)} articles!\n")
    
    for i, row in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Article {i}:")
        print(f"📅 Date: {row.date}")
        print(f"🔗 URL: {row.url}")
        print(f"📊 Tone: {row.tone}")
        print(f"📝 Preview: {row.text[:300]}...")
        print()
    
except Exception as e:
    print(f"❌ Error: {e}")