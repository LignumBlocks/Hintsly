import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

def connect_to_postgres():
    """
    Establishes a connection to the PostgreSQL database using the connection string
    from the environment variable 'DATABASE_URL'.
    
    Returns:
        psycopg2.connection: A connection object to interact with the PostgreSQL database.
    """
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def read_from_postgres(query: str):
    """
    Executes a PostgreSQL query and retrieves the result from the PostgreSQL database.

    Args:
        query (str): The PostgreSQL query to be executed.
    
    Returns:
        tuple: A tuple containing two elements:
            - rows (list): A list of rows retrieved from the query.
            - colnames (list): A list of column names from the query result.
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        # Execute the provided query
        cursor.execute(query)
        # Fetch column names from the result
        colnames = [desc[0] for desc in cursor.description]
        print(f"{' | '.join(colnames)}")
        print("-" * 80)

        # Fetch all rows
        rows = cursor.fetchall()
        # print(rows[0])
        return rows, colnames
    except Exception as e:
        print(f"Error reading from PostgreSQL: {e}")
    finally:
        cursor.close()
        conn.close()

def execute_in_postgres(query: str):
    """
    Executes an PostgreSQL statement that modifies data in the PostgreSQL database (e.g., INSERT, UPDATE, DELETE).

    Args:
        query (str): The PostgreSQL query to be executed.
    
    Returns:
        cursor (psycopg2.cursor): A cursor object for the executed query.
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        # Execute the provided query and commit the transaction
        cursor.execute(query)
        conn.commit()
        return cursor
    except Exception as e:
        print(f"Error excecuting at PostgreSQL: {e}\n{query}")
    finally:
        cursor.close()
        conn.close()

def get_table_structures():
    """
    Retrieves the structure of all tables in the 'public' schema from the PostgreSQL database,
    including table names, column names, data types, and default values.

    Returns:
        dict: A dictionary representing the table structure. 
        The keys are table names, and the values are dictionaries with column names as keys and
        data type and column default values as nested dictionaries.
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        # Query to fetch table names, column names, data types, and default values
        cursor.execute("""
            SELECT table_name, column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)
        table_structure = {}
        current_table = None
        # Process each row from the result
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
    Create a table it the PostgreSQL database if it does not exists

    Args:
        table_name (str): The name of the table to create.
        columns (dict): A dictionary defining the table's columns. 
                        The keys are column names and the values are data types (e.g., {"id": "SERIAL PRIMARY KEY", "name": "TEXT"}).
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        # Construct the column definitions for the CREATE TABLE statement
        column_defs = ", ".join(f"{name} {data_type}" for name, data_type in columns.items())
    
        # Build the CREATE TABLE query
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {column_defs}
            );
        """
        print(query)
        
        # Execute the query and commit the changes
        cursor.execute(query)
        conn.commit()
        print(f"Table '{table_name}' created.")
    except Exception as e:
        print(f"Error creating the table '{table_name}': {e}")
    finally:
        cursor.close()
        conn.close()
        
def delete_tables(table_names):
    """
    Deletes the specified tables from the PostgreSQL database if they exist.

    Args:
        table_names (list): A list of table names to delete.
    """
    conn = connect_to_postgres()
    cursor = conn.cursor()
    try:
        # Loop through the provided table names and delete each one
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
    
    # Define the tables structures
    tables = {
        "transcriptions": {
            "id": "SERIAL PRIMARY KEY",
            "download": "VARCHAR(255)",
            "url": "VARCHAR(255)",
            "content": "TEXT",
            "channel_name": "VARCHAR(255)",
            "video_identifier" : "VARCHAR(255)",
            "video_publication_date": "DATE",
            "added_at": "TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()"
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

    # table_structure = get_table_structures()

    # # Print the table structure
    # for table_name, columns in table_structure.items():
    #     print(f"Table: {table_name}")
    #     for column_name, info in columns.items():
    #         print(f"  Column: {column_name}, Type: {info['data_type']}, Default: {info['column_default']}")