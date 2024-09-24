import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

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
        conn.commit()
        return cursor
    except Exception as e:
        print(f"Error excecuting at PostgreSQL: {e}\n{query}")
    finally:
        cursor.close()
        conn.close()

def get_table_structure():
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT table_name, column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)
        table_structure = {}
        current_table = None
        for row in cursor:
            table_name = row[0]
            if current_table != table_name:
                current_table = table_name
                table_structure[table_name] = {}
            table_structure[table_name][row[1]] = {
                "data_type": row[2],
                "column_default": row[3]
            }
        return table_structure
    except Exception as e:
        print(f"Error getting table structure: {e}")
    finally:
        cursor.close()
        conn.close()

def create_table(table_name: str, columns: dict):
    """
    Create a table at the database if does not exists

    Args:
        table_name (str): Name of the table to create.
        columns (dict): Dictionary with the columns in the following format {column_name: data_type}.
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        column_defs = ", ".join(f"{name} {data_type}" for name, data_type in columns.items())
    
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {column_defs}
            );
        """
        print(query)
        
        cursor.execute(query)
        conn.commit()
        print(f"Table '{table_name}' created.")
    except Exception as e:
        print(f"Error creating the table '{table_name}': {e}")
    finally:
        cursor.close()
        conn.close()
        
def delete_tables(table_names):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        for table_name in table_names:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"Tables {table_names} deleted successfully.")
    except Exception as e:
        print(f"Error deleting tables: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # delete_tables(['hack_financial_category', "hack_complexity", "complexities", "financial_categories", "hack_classifications"])
    alter_query = """
    ALTER TABLE validation 
    RENAME COLUMN query_id TO hack_id;

    ALTER TABLE validation
    ALTER COLUMN hack_id TYPE INTEGER;

    ALTER TABLE validation
    ADD CONSTRAINT hack_id_fkey FOREIGN KEY (hack_id) REFERENCES hacks(id);"""
    alter_query = """
    ALTER TABLE validation_sources
    ADD COLUMN query TEXT;
    """
    alter_query = """ALTER TABLE hack_financial_category 
    RENAME COLUMN complexity_details TO financial_category_details;"""
    # execute_in_postgres(alter_query)
    # Define the tables structures
    tables = {
        "transcriptions": {
            "id": "SERIAL PRIMARY KEY",
            "file_name": "VARCHAR(255)",
            "source": "VARCHAR(255)",
            "content": "TEXT"
        },
        "hacks_verification": {
            "id": "SERIAL PRIMARY KEY",
            "transcription_id": "INTEGER REFERENCES transcriptions(id)",
            "hack_status": "BOOLEAN"
        },
        "hacks": {
            "id": "SERIAL PRIMARY KEY",
            "transcription_id": "INTEGER REFERENCES transcriptions(id)",
            "title": "VARCHAR(255)",
            "summary": "TEXT",
            "justification": "TEXT"
        },
        "queries": {
            "id": "SERIAL PRIMARY KEY",
            "hack_id": "INTEGER REFERENCES hacks(id)",
            "query_list": "JSONB"
        },
        "validation_sources": {
            "id": "SERIAL PRIMARY KEY",
            "query_id": "INTEGER REFERENCES queries(id)",
            "query": "TEXT",
            "source": "VARCHAR(255)",
            "title": "VARCHAR(255)",
            "description": "TEXT",
            "link": "TEXT",
            "content": "TEXT",
            "scraped_at": "TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()"
        },
        "validation": {
            "id": "SERIAL PRIMARY KEY",
            "hack_id": "INTEGER REFERENCES hacks(id)",
            "validation_status": "VARCHAR(255)",
            "validation_analysis": "TEXT",
            "relevant_sources": "TEXT"
        },
        "validated_hacks": {
            "id": "SERIAL PRIMARY KEY",
            "hack_id": "INTEGER REFERENCES hacks(id)",
            "validation_id": "INTEGER REFERENCES validation(id)"
        },
        "hack_descriptions": {
            "id": "SERIAL PRIMARY KEY",
            "hack_id": "INTEGER REFERENCES hacks(id)",
            "free_description": "TEXT",
            "premium_description": "TEXT"
        },
        "hack_structured_info": {
            "id": "SERIAL PRIMARY KEY",
            "description_id": "INTEGER REFERENCES hack_descriptions(id)",
            "hack_title": "TEXT",
            "description": "TEXT",
            "main_goal": "TEXT",
            "steps_summary": "JSONB",
            "resources_needed": "JSONB",
            "expected_benefits": "JSONB",
            "extended_title": "TEXT",
            "detailed_steps": "JSONB",
            "additional_tools_resources": "JSONB",
            "case_study": "JSONB"
        },
        "classified_hack": {
            "id": "SERIAL PRIMARY KEY",
            "hack_id": "INTEGER REFERENCES hacks(id)",
            "complexity_id": "INTEGER REFERENCES complexities(id)",
            "financial_category_id": "INTEGER REFERENCES financial_categories(id)",      
        },
        "complexities": {
            "id": "SERIAL PRIMARY KEY",
            "value": "VARCHAR(255) UNIQUE NOT NULL",
            "value_description": "TEXT"
        },
        "financial_categories": {
            "id": "SERIAL PRIMARY KEY",
            "value": "VARCHAR(255) UNIQUE NOT NULL",
            "value_description": "TEXT"
        },
        "hack_complexity": {
            "id": "SERIAL PRIMARY KEY",
            "description_id": "INTEGER REFERENCES hack_descriptions(id)",
            "complexity_id": "INTEGER REFERENCES complexities(id)",
            "complexity_details": "TEXT"
        },
        "hack_financial_category": {
            "id": "SERIAL PRIMARY KEY",
            "description_id": "INTEGER REFERENCES hack_descriptions(id)",
            "financial_category_id": "INTEGER REFERENCES financial_categories(id)",
            "financial_category_details": "TEXT"
        }
    }

    # # Create the tables
    # for table_name, columns in tables.items():
    #     create_table(table_name, columns)

    table_structure = get_table_structure()

    # Print the table structure
    for table_name, columns in table_structure.items():
        print(f"Table: {table_name}")
        for column_name, info in columns.items():
            print(f"  Column: {column_name}, Type: {info['data_type']}, Default: {info['column_default']}")