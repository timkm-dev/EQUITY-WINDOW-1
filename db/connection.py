import psycopg2
from sqlalchemy import create_engine
from config import DB_CONFIG, DATABASE_URL

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_engine():
    return create_engine(DATABASE_URL)