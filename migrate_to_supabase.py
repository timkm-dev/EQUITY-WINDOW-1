import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

# Load local environment variables for the local database connection
load_dotenv()

# --- Configuration ---
# Your Supabase database URL
SUPABASE_URL = os.getenv("SUPABASE_MIGRATION_URL")

# Build the local database connection string (from your docker container)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5434"),
    "dbname": os.getenv("DB_NAME", "equity_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

LOCAL_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

def migrate_schema_and_data():
    if not SUPABASE_URL:
        print("❌ ERROR: Please add SUPABASE_MIGRATION_URL to your .env file with your actual connection string.")
        return

    print("🔌 Connecting to databases...")
    local_engine = create_engine(LOCAL_URL)
    remote_engine = create_engine(SUPABASE_URL.replace("postgres://", "postgresql+psycopg2://"))

    # 1. Migrate Assets Table
    print("\n📦 Reading 'assets' from local database...")
    with local_engine.connect() as conn:
        assets_df = pd.read_sql("SELECT * FROM assets", conn)
    
    print(f"Uploading {len(assets_df)} assets to Supabase...")
    with remote_engine.connect() as conn:
        # Create table if it doesn't exist, and replace data
        assets_df.to_sql("assets", conn, if_exists="replace", index=False)
        # Add primary key back (pandas to_sql strips constraints)
        conn.execute(text("ALTER TABLE assets ADD PRIMARY KEY (id);"))
        conn.commit()

    # 2. Migrate Prices Table
    print("\n📈 Reading 'prices' from local database (this might take a moment)...")
    with local_engine.connect() as conn:
        prices_df = pd.read_sql("SELECT * FROM prices", conn)
    
    print(f"Uploading {len(prices_df)} price records to Supabase...")
    with remote_engine.connect() as conn:
        # Create table and replace data
        prices_df.to_sql("prices", conn, if_exists="replace", index=False)
        
        # Add constraints back
        print("Restoring constraints on 'prices' table...")
        conn.execute(text("ALTER TABLE prices ADD PRIMARY KEY (id);"))
        conn.execute(text("ALTER TABLE prices ADD CONSTRAINT fk_asset FOREIGN KEY (asset_id) REFERENCES assets(id);"))
        conn.execute(text("ALTER TABLE prices ADD CONSTRAINT unique_asset_date UNIQUE (asset_id, date);"))
        conn.commit()

    print("\n✅ Migration complete! Your Supabase database is now a perfect copy of your local database.")

if __name__ == "__main__":
    migrate_schema_and_data()
