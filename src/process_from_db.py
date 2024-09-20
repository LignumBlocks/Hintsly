from handle_db import *
from process_and_validate import discriminate_hacks_from_text, get_queries_for_validation, validate_financial_hack, get_deep_analysis
import pandas as pd
from settings import BASE_DIR, SRC_DIR, DATA_DIR

def write_to_postgres(df, table_name='validation_results'):
    execute_in_postgres()
    conn = connect_to_postgres()
    cursor = conn.cursor("""CREATE TABLE IF NOT EXISTS {table_name} (
    file_name TEXT,
    title TEXT,
    brief_summary TEXT,
    validation_status TEXT,
    validation_analysis TEXT,
    relevant_sources TEXT
);""")

    try:
        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                file_name TEXT,
                title TEXT,
                brief_summary TEXT,
                validation_status TEXT,
                validation_analysis TEXT,
                relevant_sources TEXT
            );
        """)

        # Insert data into the table
        for index, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {table_name} (
                    file_name, title, brief_summary, validation_status, validation_analysis, relevant_sources
                ) VALUES (
                    '{row['file_name']}', '{row['title']}', '{row['brief summary']}', '{row['validation status']}', 
                    '{row['validation analysis']}', '{row['relevant_sources']}'
                );
            """)
        conn.commit()  # Commit changes to the database
        print(f"Successfully inserted {len(df)} rows into {table_name}")

    except Exception as e:
        print(f"Error writing to PostgreSQL: {e}")
    finally:
        cursor.close()
        conn.close()
def validate_hacks(hacks_queries_csv_path: str):
    def get_clean_links(metadata):
        links = [item[0] for item in metadata]
        unique_links = set(links)
        result_string = ' '.join(unique_links)
        return result_string
    output_path = os.path.join(DATA_DIR, 'validation')
    if os.path.exists(output_path) and os.listdir(output_path):
        # Load existing dataframes
        dataframes = []
        for filename in os.listdir(output_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(output_path, filename)
                df = pd.read_csv(filepath)
                dataframes.append(df)
    else: 
        # Read from the database
        rows, colnames = read_from_postgres() 
        dataframes = []
        file_names = sorted(set([row[1] for row in rows]))  # Get unique file names
        for file_name in file_names:
            df_data = []
            for row in rows:
                if row[1] == file_name:
                    # Extract the desired columns for the DataFrame
                    selected_data = [row[i] for i in [1, 3, 2, 4, 5, 6, 7]]  # query, source, title, description, link, content
                    df_data.append(selected_data)
            df = pd.DataFrame(df_data, columns=['file_name', 'query', 'source', 'title', 'description', 'link', 'content'])
            dataframes.append(df)
    
        for i, df in enumerate(dataframes):
            file_name = f"sources_for_validation_dataframe_{i}.csv"
            filepath = os.path.join(output_path, file_name)
            df.to_csv(filepath, index=False)
    
    hacks_queries_df = pd.read_csv(hacks_queries_csv_path) # file_name, title, brief summary, queries
    validation_result_csv_path = os.path.join(DATA_DIR, 'validation_result_test1.csv')  # Assuming you still need this for testing?
    if os.path.isfile(validation_result_csv_path):
        df = pd.read_csv(validation_result_csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'title', 'brief summary', 'validation status', 'validation analysis', 'relevant sources'])
    
    counter = 0

    for index, row in hacks_queries_df.iterrows():
        file_name = row['file_name']
        title = row['title']
        brief_summary = row['brief summary']

        assert dataframes[index]['file_name'].iloc[0] == file_name

        results, prompt, metadata = validate_financial_hack(file_name, title, brief_summary, dataframes[index])
        status = results['validation status']
        analysis = results['validation analysis']
        relevant_sources = get_clean_links(metadata)
        print(title,':\n',analysis,'\n',status,'\n',relevant_sources)
        new_row_index = len(df)
        df.loc[new_row_index] = [file_name, title, brief_summary, status, analysis, relevant_sources]
        counter += 1

        if counter % 5 == 0:
            df.to_csv(validation_result_csv_path, index=False)
            print(f'Saved {counter} files to CSV.')

    # Save any remaining data
    if not df.empty:
        df.to_csv(validation_result_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')
