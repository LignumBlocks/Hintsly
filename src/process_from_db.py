import os
import random
from handle_db import read_from_postgres, execute_in_postgres
from process_and_validate import (verify_hacks_from_text, get_queries_for_validation, validate_financial_hack, 
                                  get_deep_analysis, enriched_analysis, get_structured_analysis, get_hack_classifications)
from settings import BASE_DIR, SRC_DIR, DATA_DIR
import llm_models

def hacks_verification():
    """
    Verifies financial hacks from transcriptions that have not been processed yet.
    For each unprocessed transcription:
    - It checks if the transcription content contains a hack.
    - If classified as a hack, it inserts the relevant data into the 'hacks_verification' and 'hacks' tables in the database.

    The verification process involves:
    - Querying the database to get unprocessed transcriptions.
    - Using an AI model to determine if the transcription constitutes a hack.
    - Inserting the hack status into the 'hacks_verification' table.
    - If it's determined to be a hack, inserting the hack details into the 'hacks' table.
    """
    # Query to retrieve all transcriptions that haven't been verified 
    unprocessed_transcriptions_query = """
    SELECT id, content
    FROM transcriptions
    WHERE id NOT IN (SELECT transcription_id FROM hacks_verification);"""
    unprocessed_transcriptions, _ = read_from_postgres(unprocessed_transcriptions_query)

    # Process each unprocessed transcription
    for transcription in unprocessed_transcriptions:
        transcription_id = transcription[0]
        text_content = transcription[1]

        # Verify if the transcription contains a hack
        result, prompt = verify_hacks_from_text(text_content)
        
        # Insert the verification result into the 'hacks_verification' table
        insert_hack_verification_query = f"""
        INSERT INTO hacks_verification (transcription_id, hack_status)
        VALUES ({transcription_id}, {result['is_a_hack']});"""
        
        execute_in_postgres(insert_hack_verification_query)

        # If the text is classified as a hack, insert it into the 'hacks' table
        if result['is_a_hack'] == True:
            insert_hack_query = f"""
            INSERT INTO hacks (transcription_id, title, summary, justification)
            VALUES ({transcription_id}, {result['possible hack title']}, {result['brief summary']}, {result['justification']});"""
            execute_in_postgres(insert_hack_query)

    print(f"Verified successfully {len(unprocessed_transcriptions)} hacks from transcriptions.")

def get_queries():
    """
    Generates validation queries for hacks that have not yet been associated with queries in the database.
    For each hack without queries:
    - Generates a list of validation queries using an AI model.
    - Inserts the generated queries into the 'queries' table.

    The process involves:
    - Querying the database for hacks that are missing queries.
    - Using an AI model to generate validation queries based on the hack title and summary.
    - Inserting the generated queries into the 'queries' table.
    """
    # Query to fetch hacks that don't have associated validation queries
    hacks_query = """
    SELECT id, title, summary
    FROM hacks
    WHERE id NOT IN (SELECT hack_id FROM queries);"""
    hacks, _ = read_from_postgres(hacks_query)

    # Process each hack and generate queries
    for hack in hacks:
        hack_id = hack[0]
        title = hack[1]
        summary = hack[2]

        # Generate validation queries for the hack using the AI model
        query_results, prompt = get_queries_for_validation(title, summary)
        query_results = query_results['queries']

        # Insert the generated queries into the 'queries' table
        insert_queries_query = f"""
        INSERT INTO queries (hack_id, query_list)
        VALUES ({hack_id}, {query_results});"""
        execute_in_postgres(insert_queries_query, (hack_id, query_results))

    print(f"Queries ready for {len(hacks)} hacks.")

def validate_hacks():
    """
    Validates financial hacks by using validation sources, analyzing them with an AI model,
    and then storing the validation results into the database.
    
    The process involves:
    - Fetching unvalidated hacks from the 'hacks' table.
    - Retrieving relevant validation sources for each hack from the 'validation_sources' table.
    - Validating each hack using an AI model, which analyzes the provided sources.
    - Inserting the validation results into the 'validation' table.
    - If the hack is considered valid, inserting it into the 'validated_hacks' table.
    """
    def get_clean_links(metadata):
        """
        Extracts non-repeated links from metadata and returns them as a single concatenated string.

        Args:
            metadata (list): A list of metadata dictionaries where each contains 'link' as the first item.
        
        Returns:
            str: A space-separated string of unique links.
        """
        links = [item[0] for item in metadata]
        unique_links = set(links)
        result_string = ' '.join(unique_links)
        return result_string
    
    # Query to get all hacks that have not been validated yet
    unvalidated_hacks_query = """
    SELECT h.id, h.title, h.summary
    FROM hacks h
    LEFT JOIN validation v ON h.id = v.hack_id
    WHERE v.hack_id IS NULL;"""
    unvalidated_hacks, _ = read_from_postgres(unvalidated_hacks_query)

    # Query to get validation sources related to the unvalidated hacks
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
    
    # Process each unvalidated hack for validation
    for hack in unvalidated_hacks:
        hack_id = hack[0]
        title = hack[1]
        brief_summary = hack[2]

         # Retrieve validation sources for the current hack
        validation_sources_for_hack = validation_sources_by_hack.get(hack_id, [])

        # Validate the hack using an AI model and get the results
        results, prompt, metadata = validate_financial_hack(str(hack_id), title, brief_summary, validation_sources_for_hack)
        status = results['validation status']
        analysis = results['validation analysis']
        relevant_sources = get_clean_links(metadata)

        # Insert validation result into 'validation' table
        insert_validation_query = f"""
        INSERT INTO validation (hack_id, validation_status, validation_analysis, relevant_sources)
        VALUES ({hack_id}, {status}, {analysis}, {relevant_sources});"""
        execute_in_postgres(insert_validation_query)

        # Insert into 'validated_hacks' table if valid# If the hack is valid, insert it into the 'validated_hacks' table
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
    """
    Analyzes validated hacks that are not yet in the 'hack_descriptions' table. The process includes:
    - Fetching validated hacks that lack detailed descriptions.
    - Performing deep analysis (both free and premium) on each hack's content.
    - Enriching the analysis results with additional context.
    - Storing the deep analysis in the 'hack_descriptions' table.
    - Structuring the detailed analysis and storing it in the 'hack_structured_info' table.
    """
    # Query to fetch validated hacks that have not been analyzed
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
    
    # Process each unanalized hack
    for hack in unanalized_hacks:
        hack_id = hack[0]
        title = hack[1]
        summary = hack[2]
        content = hack[3]

        # Get deep analysis results (free and premium) for the hack
        result_free, result_premium, structured_free, structured_premium = get_deep_analysis(title, summary, content)

        # Enrich the descriptions using validation sources
        new_result_free, new_result_premium = grow_descriptions(hack_id, result_free, result_premium)

        # Save the analysis results in the 'hack_descriptions' table
        description_query = f"""
            INSERT INTO hack_descriptions (hack_id, title, brief_summary, deep_analysis_free, deep_analysis_premium)
            VALUES ({hack_id}, {title}, {summary}, {result_free}, {result_premium});"""
        execute_in_postgres(description_query)

        # Get structured dictionary of the descriptions (free and premium) after enrichment
        structured_free, structured_premium, _, _ = get_structured_analysis(new_result_free, new_result_premium)
        
        # Save structured analysis in the 'hack_structured_info' table
        hack_title = structured_free["Hack Title"]
        description = structured_free["Description"]
        main_goal = structured_free["Main Goal"]
        steps_summary = structured_free["steps(Summary)"]
        resources_needed = structured_free["Resources Needed"]
        expected_benefits = structured_free["Expected Benefits"]
        extended_title = structured_premium["Extended Title"]
        detailed_steps = structured_premium["Detailed steps"]
        additional_tools_resources = structured_premium["Additional Tools and Resource"]
        case_study = structured_premium["Case Study"]
        
        structured_query = f"""
            INSERT INTO hack_structured_info (description_id, hack_title, description, main_goal, steps_summary, resources_needed, 
                    expected_benefits, extended_title, detailed_steps, additional_tools_resources, case_study)
            VALUES ((SELECT id FROM hack_descriptions WHERE hack_id = {hack_id} ORDER BY id DESC LIMIT 1)), 
                {hack_title}, {description}, {main_goal}, {steps_summary}, {resources_needed}, {expected_benefits}, 
                {extended_title}, {detailed_steps}, {additional_tools_resources}, {case_study});"""
        execute_in_postgres(structured_query)

    print(f"Completed descriptions for {len(unanalized_hacks)} hacks.")

def grow_descriptions(hack_id, free_description, premium_description, times=4, k=5):
    """
    Enriches the initial free and premium analysis of a hack by iterating through multiple document chunks
    retrieved from a similarity search in the vector store. This process extends the existing descriptions
    with relevant context.

    Args:
        hack_id (str): The ID of the hack.
        free_description (str): The initial free analysis of the hack.
        premium_description (str): The initial premium analysis of the hack.
        times (int): The number of iterations to extend the descriptions (default is 4).
        k (int): The number of documents to use in each iteration (default is 5).

    Returns:
        tuple: A tuple containing the enriched free and premium descriptions.
    """
    rag = llm_models.RAG_LLMmodel("gpt-4o-mini", chroma_path=os.path.join(DATA_DIR, 'chroma_db'))
    
    # Retrieve documents similar to the hack descriptions
    documents = rag.retrieve_similar_for_hack(hack_id, free_description+premium_description, k=k*times)
    # Randomize the order of elements in the list
    random.shuffle(documents)

    latest_free = free_description
    latest_premium = premium_description

    # Extend the descriptions using document chunks
    for i in range(times):
        chunks = ""
        for document in documents[i * k: (i + 1) * k]:
            chunks += f"Relevant context section: \n\"\"\"{document.page_content}\n\"\"\" \n"
        print(f"Extending descriptions: iter = {i+1}")
        free_description, premium_description, _, _= enriched_analysis(latest_free, latest_premium, chunks)
        latest_free = free_description
        latest_premium = premium_description
    return latest_free, latest_premium

def classify_hacks():
    """
    Classifies hacks a financial hack based on several parameters. The process includes:
    - Retrieving hacks that have not been classified.
    - Using an AI model to assign a classification to the hack for each parameter.
    - Inserting the classification results into the corresonding tables.
    """
    # Query to retrieve hacks that have not been classified yet
    unprocessed_hacks_query = """
    SELECT h.id, h.title, h.summary, hd.free_description, hd.id
    FROM hacks h 
    JOIN hack_descriptions hd ON h.id = hd.hack_id 
    WHERE NOT EXISTS (SELECT 1 FROM classified_hack WHERE classified_hack.hack_id = h.id);"""
    unprocessed_hacks, _ = read_from_postgres(unprocessed_hacks_query)

    # Process each hack for classification
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

        # Retrieve the ID of the complexity classification value
        complexity_query = f"SELECT id FROM complexities WHERE value = '{complexity_class}';"
        complexity_id, _ = read_from_postgres(complexity_query)
        complexity_id = complexity_id[0][0] if complexity_id else None

        # Get financial categories and their explanations
        financial_categories = [(item["category"], item["breve explanation"]) for item in result_categories]
        
        # Retrieve the IDs of the financial category values
        financial_category_ids = []
        for category, details in financial_categories:
            category_query = f"SELECT id FROM financial_categories WHERE value = '{category}';"
            category_id, _ = read_from_postgres(category_query)
            if category_id:
                financial_category_ids.append((category_id[0][0], details))

        # Insert the complexity classification into 'hack_complexity' table
        if complexity_id is not None:
            complexity_insert_query = f"""
            INSERT INTO hack_complexity (description_id, complexity_id, complexity_details) 
            VALUES ({description_id}, {complexity_id}, {complexity_details});
            """
            execute_in_postgres(complexity_insert_query)

        # Insert each financial category into the 'hack_financial_category' table
        for financial_category_id, financial_category_details in financial_category_ids:
            financial_category_insert_query = f"""
            INSERT INTO hack_financial_category (description_id, financial_category_id, financial_category_details) 
            VALUES ({hack_id}, {financial_category_id}, {financial_category_details});"""
            execute_in_postgres(financial_category_insert_query)

    print(f"Completed classification for {len(unprocessed_hacks)} hacks.")
