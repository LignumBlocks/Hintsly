import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

# Función para conectarse a PostgreSQL
def connect_to_postgres():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def read_from_postgres(query: str = "SELECT * FROM scraping_results;"):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]
        print(f"{' | '.join(colnames)}") # id | file_name | source | query | title | description | link | content | scraped_at
        print("-" * 80)

        rows = cursor.fetchall()
        print(rows[0])
        return rows, colnames
    except Exception as e:
        print(f"Error reading from PostgreSQL: {e}")
    finally:
        cursor.close()
        conn.close()

def execute_in_postgres(query: str):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        print(cursor)
        return cursor
    except Exception as e:
        print(f"Error excecuting at PostgreSQL: {e}\n{query}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    read_from_postgres()