"""
Test GDELT BigQuery fetcher - CORRECTED COLUMN NAMES
"""
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

# CORRECTED QUERY - using actual GDELT column names
query = """
SELECT 
    DATE(_PARTITIONTIME) as date,
    DocumentIdentifier as url,
    V2Tone as tone,
    V2Themes as themes,
    V2Locations as locations,
    V2Persons as persons,
    V2Organizations as organizations,
    SourceCollectionIdentifier as source
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE 
    _PARTITIONTIME >= '2024-01-01'
    AND _PARTITIONTIME <= '2024-12-31'
    AND (
        LOWER(V2Locations) LIKE '%calabarzon%'
        OR LOWER(V2Locations) LIKE '%batangas%'
        OR LOWER(V2Locations) LIKE '%laguna%'
        OR LOWER(V2Locations) LIKE '%cavite%'
        OR LOWER(V2Locations) LIKE '%rizal%'
        OR LOWER(V2Locations) LIKE '%quezon%'
        OR LOWER(V2Themes) LIKE '%food%'
        OR LOWER(V2Themes) LIKE '%hunger%'
        OR LOWER(V2Themes) LIKE '%rice%'
    )
LIMIT 10
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
        print(f"📍 Locations: {row.locations}")
        print(f"🎯 Themes: {row.themes}")
        print()
    
    if len(results) == 0:
        print("No articles found for CALABARZON specific. Let me try a broader query...")
        
        # Broader query for Philippines food news
        query2 = """
        SELECT 
            DATE(_PARTITIONTIME) as date,
            DocumentIdentifier as url,
            V2Tone as tone,
            V2Themes as themes,
            V2Locations as locations
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE 
            _PARTITIONTIME >= '2024-01-01'
            AND _PARTITIONTIME <= '2024-12-31'
            AND LOWER(V2Themes) LIKE '%food%'
        LIMIT 10
        """
        
        results2 = list(client.query(query2).result())
        print(f"\n✅ Found {len(results2)} articles about food in general!\n")
        
        for i, row in enumerate(results2, 1):
            print(f"{'='*60}")
            print(f"Article {i}:")
            print(f"📅 Date: {row.date}")
            print(f"🔗 URL: {row.url}")
            print(f"📊 Tone: {row.tone}")
            print(f"📍 Locations: {row.locations}")
            print(f"🎯 Themes: {row.themes}")
            print()
    
except Exception as e:
    print(f"❌ Error: {e}")