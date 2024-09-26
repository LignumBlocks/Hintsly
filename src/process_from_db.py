import pandas as pd
import random
from handle_db import *
from process_and_validate import (verify_hacks_from_text, get_queries_for_validation, validate_financial_hack, 
                                  get_deep_analysis, enriched_analysis, get_structured_analysis, get_hack_classifications)
from settings import BASE_DIR, SRC_DIR, DATA_DIR
import llm_models

def hacks_verification():
    # Get all transcriptions that haven't been processed
    unprocessed_transcriptions_query = """
    SELECT id, content
    FROM transcriptions
    WHERE id NOT IN (SELECT transcription_id FROM hacks_verification);"""
    unprocessed_transcriptions, _ = read_from_postgres(unprocessed_transcriptions_query)

    for transcription in unprocessed_transcriptions:
        transcription_id = transcription[0]
        text_content = transcription[1]

        # Process the text content
        result, prompt = verify_hacks_from_text(text_content)
        
        # Insert into hacks_verification
        insert_hack_verification_query = f"""
        INSERT INTO hacks_verification (transcription_id, hack_status)
        VALUES ({transcription_id}, {result['is_a_hack']});"""
        
        execute_in_postgres(insert_hack_verification_query)
        
        if result['is_a_hack'] == True:
            # Insert into hacks if is_a_hack
            insert_hack_query = f"""
            INSERT INTO hacks (transcription_id, title, summary, justification)
            VALUES ({transcription_id}, {result['possible hack title']}, {result['brief summary']}, {result['justification']});"""
            execute_in_postgres(insert_hack_query)

    print(f"Verified successfully {len(unprocessed_transcriptions)} hacks from transcriptions.")

def get_queries():
    # Get hack information from the 'hacks' table that are not in 'queries'
    hacks_query = """
    SELECT id, title, summary
    FROM hacks
    WHERE id NOT IN (SELECT hack_id FROM queries);"""
    hacks, _ = read_from_postgres(hacks_query)

    for hack in hacks:
        hack_id = hack[0]
        title = hack[1]
        summary = hack[2]

        # Generate queries
        query_results, prompt = get_queries_for_validation(title, summary)
        query_results = query_results['queries']

        # Insert into 'queries' table
        insert_queries_query = f"""
        INSERT INTO queries (hack_id, query_list)
        VALUES ({hack_id}, {query_results});"""
        execute_in_postgres(insert_queries_query, (hack_id, query_results))

    print(f"Queries ready for {len(hacks)} hacks.")

def validate_hacks():
    def get_clean_links(metadata):
        links = [item[0] for item in metadata]
        unique_links = set(links)
        result_string = ' '.join(unique_links)
        return result_string
    
    # Get unvalidated hacks from 'hacks' table
    unvalidated_hacks_query = """
    SELECT h.id, h.title, h.summary
    FROM hacks h
    LEFT JOIN validation v ON h.id = v.hack_id
    WHERE v.hack_id IS NULL;"""
    unvalidated_hacks, _ = read_from_postgres(unvalidated_hacks_query)
    # Get validation sources related to unvalidated hacks
    validation_sources_query = """
    SELECT q.hack_id, vs.query, vs.source, vs.title, vs.description, vs.link, vs.content
    FROM queries q
    JOIN validation_sources vs ON q.id = vs.query_id
    WHERE q.hack_id IN (
        SELECT id
        FROM hacks
        WHERE id NOT IN (SELECT hack_id FROM validation)
    );"""
    validation_sources, _ = read_from_postgres(validation_sources_query)

    # Group validation sources by hack_id
    validation_sources_by_hack = {}
    for row in validation_sources:
        hack_id = row[0]
        if hack_id not in validation_sources_by_hack:
            validation_sources_by_hack[hack_id] = []

        validation_sources_by_hack[hack_id].append({
            "query": row[1],
            "source": row[2],
            "title": row[3],
            "description": row[4],
            "link": row[5],
            "content": row[6]
        })
    
    # Process hacks for validation
    for hack in unvalidated_hacks:
        hack_id = hack[0]
        title = hack[1]
        brief_summary = hack[2]

        # Get validation sources for the hack
        validation_sources_for_hack = validation_sources_by_hack.get(hack_id, [])

        # Validate the hack
        results, prompt, metadata = validate_financial_hack(str(hack_id), title, brief_summary, validation_sources_for_hack)
        status = results['validation status']
        analysis = results['validation analysis']
        relevant_sources = get_clean_links(metadata)

        # Insert validation result into 'validation' table
        insert_validation_query = f"""
        INSERT INTO validation (hack_id, validation_status, validation_analysis, relevant_sources)
        VALUES ({hack_id},{status}, {analysis}, {relevant_sources});"""
        execute_in_postgres(insert_validation_query)

        # Insert into 'validated_hacks' table if valid
        if status == 'Valid':
            insert_validated_hacks_query = f"""
            INSERT INTO validated_hacks (hack_id, validation_id)
            VALUES ({hack_id}, (SELECT id FROM validation WHERE hack_id = {hack_id} ORDER BY id DESC LIMIT 1));"""
            execute_in_postgres(insert_validated_hacks_query)

    print(f"Completed validation for {len(unvalidated_hacks)} hacks.")
    # output_path = os.path.join(DATA_DIR, 'validation')
    # if os.path.exists(output_path) and os.listdir(output_path):
    #     # Load existing dataframes
    #     dataframes = []
    #     for filename in os.listdir(output_path):
    #         if filename.endswith(".csv"):
    #             filepath = os.path.join(output_path, filename)
    #             df = pd.read_csv(filepath)
    #             dataframes.append(df)
    # else: 
    #     # Read from the database
    #     rows, colnames = read_from_postgres() 
    #     dataframes = []
    #     file_names = sorted(set([row[1] for row in rows]))  # Get unique file names
    #     for file_name in file_names:
    #         df_data = []
    #         for row in rows:
    #             if row[1] == file_name:
    #                 # Extract the desired columns for the DataFrame
    #                 selected_data = [row[i] for i in [1, 3, 2, 4, 5, 6, 7]]  # query, source, title, description, link, content
    #                 df_data.append(selected_data)
    #         df = pd.DataFrame(df_data, columns=['file_name', 'query', 'source', 'title', 'description', 'link', 'content'])
    #         dataframes.append(df)
    
    #     for i, df in enumerate(dataframes):
    #         file_name = f"sources_for_validation_dataframe_{i}.csv"
    #         filepath = os.path.join(output_path, file_name)
    #         df.to_csv(filepath, index=False)
    
    # hacks_queries_df = pd.read_csv(hacks_queries_csv_path) # file_name, title, brief summary, queries
    # validation_result_csv_path = os.path.join(DATA_DIR, 'validation_result_test1.csv')  # Assuming you still need this for testing?
    # if os.path.isfile(validation_result_csv_path):
    #     df = pd.read_csv(validation_result_csv_path)
    # else:
    #     df = pd.DataFrame(columns=['file_name', 'title', 'brief summary', 'validation status', 'validation analysis', 'relevant sources'])
    
    # counter = 0

    # for index, row in hacks_queries_df.iterrows():
    #     file_name = row['file_name']
    #     title = row['title']
    #     brief_summary = row['brief summary']

    #     assert dataframes[index]['file_name'].iloc[0] == file_name

    #     results, prompt, metadata = validate_financial_hack(file_name, title, brief_summary, dataframes[index])
    #     status = results['validation status']
    #     analysis = results['validation analysis']
    #     relevant_sources = get_clean_links(metadata)
    #     print(title,':\n',analysis,'\n',status,'\n',relevant_sources)
    #     new_row_index = len(df)
    #     df.loc[new_row_index] = [file_name, title, brief_summary, status, analysis, relevant_sources]
    #     counter += 1

    #     if counter % 5 == 0:
    #         df.to_csv(validation_result_csv_path, index=False)
    #         print(f'Saved {counter} files to CSV.')

    # # Save any remaining data
    # if not df.empty:
    #     df.to_csv(validation_result_csv_path, index=False)
    #     print('Final save: saved remaining files to CSV.')

def analyze_validated_hacks():
    # Get validated hacks that are not in the hack_descriptions table
    unanalized_hacks_query = """
        SELECT h.id, h.title, h.summary, t.content
        FROM hacks h
        JOIN validated_hacks vh ON h.id = vh.hack_id
        JOIN transcriptions t ON h.transcriptions = t.id
        WHERE NOT EXISTS (
            SELECT 1 
            FROM hack_descriptions hd 
            WHERE hd.hack_id = h.id
        );"""
    unanalized_hacks, _ = read_from_postgres(unanalized_hacks_query)
    
    for hack in [1]:
        hack_id = hack[0]
        title = hack[1]
        summary = hack[2]
        content = hack[3]

        # Get deep analysis results
        result_free, result_premium, structured_free, structured_premium = get_deep_analysis(title, summary, content)

        # Enrich it using the validation sources
        new_result_free, new_result_premium = grow_descriptions(hack_id, result_free, result_premium)

        # Save analysis results to hack_descriptions
        description_query = f"""
            INSERT INTO hack_descriptions (hack_id, title, brief_summary, deep_analysis_free, deep_analysis_premium)
            VALUES ({hack_id}, {title}, {summary}, {result_free}, {result_premium});"""
        execute_in_postgres(description_query)

        structured_free, structured_premium, _, _ = get_structured_analysis(hack_id, new_result_free, new_result_premium)
        
        # Save structured analysis to hack_structured_info
        hack_title = structured_free
        description = structured_free
        main_goal = structured_free
        steps_summary = structured_free
        resources_needed = structured_free
        expected_benefits = structured_free
        extended_title = structured_premium
        detailed_steps = structured_premium
        additional_tools_resources = structured_premium
        case_study = structured_premium
        
        structured_query = f"""
            INSERT INTO hack_structured_info (description_id, hack_title, description, main_goal, steps_summary, resources_needed, 
                    expected_benefits, extended_title, detailed_steps, additional_tools_resources, case_study)
            VALUES ((SELECT id FROM hack_descriptions WHERE hack_id = {hack_id} ORDER BY id DESC LIMIT 1)), 
                {hack_title}, {description}, {main_goal}, {steps_summary}, {resources_needed},
                {expected_benefits}, {extended_title}, {detailed_steps}, {additional_tools_resources}, {case_study}
            );"""
        execute_in_postgres(structured_query)

    print(f"Completed descriptions for {len(unanalized_hacks)} hacks.")

def grow_descriptions(hack_id, free_description, premium_description, times=4, k=5):
    model = llm_models.LLMmodel("gpt-4o-mini")
    rag = llm_models.RAG_LLMmodel("gpt-4o-mini", chroma_path=os.path.join(DATA_DIR, 'chroma_db'))
    
    documents = rag.retrieve_similar_for_hack(hack_id, free_description+premium_description, k=k*times)
    # Randomize the order of elements in the list
    random.shuffle(documents)

    for i in range(times):
        chunks = ""
        for document in documents[i * k: (i + 1) * k]:
            chunks += f"Relevant context section: \n\"\"\"{document.page_content}\n\"\"\" \n"
        print(f"Extending descriptions: iter = {i+1}")
        free_description, premium_description, _, _= enriched_analysis(free_description, premium_description, chunks)
    return free_description, premium_description

def classify_hacks():
    # Retrieve hacks not yet classified
    unprocessed_hacks_query = """
    SELECT h.id, h.title, h.summary, hd.free_description, hd.id
    FROM hacks h 
    JOIN hack_descriptions hd ON h.id = hd.hack_id 
    WHERE NOT EXISTS (SELECT 1 FROM classified_hack WHERE classified_hack.hack_id = h.id);"""
    
    unprocessed_hacks, _ = read_from_postgres(unprocessed_hacks_query)

    for hack_data in unprocessed_hacks:
        hack_id = hack_data[0]
        title = hack_data[1]
        summary = hack_data[2]
        description = hack_data[3]
        description_id = hack_data[4]
        
        # Get classification results
        result_complexity, result_categories, _, _ = get_hack_classifications(description)
        
        complexity_class = result_complexity['complexity']['classification']
        complexity_details = result_complexity['complexity']['explanation']

        # Retrieve complexity ID
        complexity_query = f"SELECT id FROM complexities WHERE value = '{complexity_class}';"
        complexity_id, _ = read_from_postgres(complexity_query)
        complexity_id = complexity_id[0][0] if complexity_id else None

        financial_categories = [(item["category"], item["breve explanation"]) for item in result_categories]
        
        # Retrieve financial category IDs
        financial_category_ids = []
        for category, details in financial_categories:
            category_query = f"SELECT id FROM financial_categories WHERE value = '{category}';"
            category_id, _ = read_from_postgres(category_query)
            if category_id:
                financial_category_ids.append((category_id[0][0], details))

        # Insert into hack_complexity table (if complexity is available)
        if complexity_id is not None:
            complexity_insert_query = f"""
            INSERT INTO hack_complexity (description_id, complexity_id, complexity_details) 
            VALUES ({description_id}, {complexity_id}, {complexity_details});
            """
            execute_in_postgres(complexity_insert_query)

        # Insert into hack_financial_category table (for each financial category)
        for financial_category_id, financial_category_details in financial_category_ids:
            financial_category_insert_query = f"""
            INSERT INTO hack_financial_category (description_id, financial_category_id, financial_category_details) 
            VALUES ({hack_id}, {financial_category_id}, {financial_category_details});"""
            execute_in_postgres(financial_category_insert_query)

    print(f"Completed classification for {len(unprocessed_hacks)} hacks.")
