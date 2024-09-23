import os
import pandas as pd
import json
from settings import BASE_DIR, SRC_DIR, DATA_DIR
from process_and_validate import discriminate_hacks_from_text, get_queries_for_validation, validate_financial_hack, get_deep_analysis, get_structured_analysis, get_hack_classifications
from process_from_db import validate_hacks

def process_transcriptions_from_csv():
    # data_folder = os.path.join(DATA_DIR, 'Transcriptions Nobudgetbabe')
    # hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination.csv')
    data_folder = os.path.join(DATA_DIR, 'test_cases')
    hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination_tests.csv')
    
    if os.path.isfile(hacks_discrimination_csv_path):
        hacks_discrimination = pd.read_csv(hacks_discrimination_csv_path)
    else:
        hacks_discrimination = pd.DataFrame(columns=['file_name', 'source', 'hack_status', 'title', 'brief summary', 'justification'])
    
    file_counter = 0
    
    # Iterate over each file in the directory in alphabetical order
    txt_files = sorted([file_name for file_name in os.listdir(data_folder) if file_name.endswith('.txt')])

    for file_name in txt_files:
        file_path = os.path.join(data_folder, file_name)
        #  Check if the file has already been processed
        file_name_without_extension = os.path.splitext(file_name)[0]
        if not hacks_discrimination[hacks_discrimination['file_name'] == file_name_without_extension].empty:
            print(f'File {file_name} already processed, skipping.')
            continue

        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        print(file_name_without_extension)
        # Process the text content
        result, prompt = discriminate_hacks_from_text(text_content, 'tik tok video')
        
        # with open(os.path.join(PROMPT_DIR, file_name), 'w') as file:
        #     file.write(prompt)

        # Prepare the new row
        new_row_index = len(hacks_discrimination)  # Get the next index
        source = (file_name_without_extension.split('_')[0])[1:]
        hacks_discrimination.loc[new_row_index] = [file_name_without_extension, source, result['is_a_hack'], result['possible hack title'], result['brief summary'], result['justification']]
        file_counter += 1
        
        # Save to CSV every 10 files
        if file_counter % 10 == 0:
            hacks_discrimination.to_csv(hacks_discrimination_csv_path, index=False)
            print(f'Saved {file_counter} files to CSV.')
    
    # Save any remaining data
    if not hacks_discrimination.empty:
        hacks_discrimination.to_csv(hacks_discrimination_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')

def get_queries_from_csv(csv_path: str):
    source_df = pd.read_csv(csv_path)
    validation_result_csv_path = os.path.join(DATA_DIR, 'validation_queries_test.csv')
    if os.path.isfile(validation_result_csv_path):
        df = pd.read_csv(validation_result_csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'title', 'brief summary', 'queries'])
    
    counter = 0

    for index, row in source_df.iterrows():
        if not row['hack_status']:
            continue
        file_name = row['file_name']     
        title = row['title']        
        brief_summary = row['brief summary'] 

        query_results, prompt = get_queries_for_validation(title, brief_summary)
        query_results = query_results['queries']
        new_row_index = len(df)
        df.loc[new_row_index] = [file_name, title, brief_summary, query_results]
        counter += 1

        if counter % 10 == 0:
            df.to_csv(validation_result_csv_path, index=False)
            print(f'Saved {counter} files to CSV.')
    # Save any remaining data
    if not df.empty:
        df.to_csv(validation_result_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')

def validate_hacks_from_csv(hacks_queries_csv_path: str, validation_sources_csvs: list):
    def get_clean_links(metadata):
        links = [item[0] for item in metadata]
        unique_links = set(links)
        result_string = ' '.join(unique_links)
        return result_string

    hacks_queries_df = pd.read_csv(hacks_queries_csv_path) # file_name, title, brief summary, queries
    validation_result_csv_path = os.path.join(DATA_DIR, 'validation_result_test1.csv')
    if os.path.isfile(validation_result_csv_path):
        df = pd.read_csv(validation_result_csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'title', 'brief summary', 'validation status', 'validation analysis', 'relevant sources'])
    
    counter = 0

    for index, row in hacks_queries_df.iterrows():
        
        file_name = row['file_name']     
        title = row['title']        
        brief_summary = row['brief summary'] 

        results, prompt, metadata = validate_financial_hack(file_name, title, brief_summary, validation_sources_csvs[index])
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

def analyze_validated_hacks_from_csv(validation_result_csv: str):
    validation_df = pd.read_csv(validation_result_csv)
    deep_analysis_csv_path = os.path.join(DATA_DIR, 'deep_analysis_results_test.csv')
    structured_deep_analysis_csv_path = os.path.join(DATA_DIR, 'structured_deep_analysis_results_test.csv')
    data_dir = os.path.join(DATA_DIR, 'test_cases')
    # if os.path.isfile(deep_analysis_csv_path):
    #     df = pd.read_csv(deep_analysis_csv_path)
    # else:
    df = pd.DataFrame(columns=['file_name', 'hack title', 'brief summary', 'deep analysis_free', 'deep analysis_premium'])
    structured_df = []

    counter = 0

    for index, row in validation_df.iterrows():
        if row['validation status'] != "Valid":
            print(row['validation status'])
            continue
        file_name = row['file_name']     
        title = row['title']        
        brief_summary = row['brief summary'] 
        file_path = os.path.join(data_dir, file_name+'.txt')
        with open(file_path, 'r') as file:
            text = file.read()

        result_free, result_premium, prompt_free, prompt_premium = get_deep_analysis(title, brief_summary, text)
        structured_free, structured_premium, _, _ = get_structured_analysis(result_free, result_premium)
        json_structured_free = json.loads(structured_free)
        json_structured_premium = json.loads(structured_premium)
        combined_data = {**json_structured_free, **json_structured_premium}
        print('combined_data', combined_data)

        new_row_index = len(df)
        df.loc[new_row_index] = [file_name, title, brief_summary, result_free, result_premium]
        
        # df_row = pd.DataFrame([combined_data]).transpose().reset_index()
        # df_row.columns = ['Column', 'Value']
        structured_df.append(combined_data)
        # print('structured_df', structured_df)
        # structured_df = pd.concat([structured_df, df_row], ignore_index=True)
        counter += 1

        if counter % 10 == 0:
            df.to_csv(deep_analysis_csv_path, index=False)
            pd.DataFrame(structured_df).fillna('N/A').to_csv(structured_deep_analysis_csv_path, index=False)
            print(f'Saved {counter} files to CSV.')
    # Save any remaining data
    if not df.empty:
        df.to_csv(deep_analysis_csv_path, index=False)
        structured_df = pd.DataFrame(structured_df).fillna('N/A')
        structured_df.to_csv(structured_deep_analysis_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')

def tag_hacks_from_csv(deep_analysis_csv_path):
    description_df = pd.read_csv(deep_analysis_csv_path)
    hacks_tagging_csv_path = os.path.join(DATA_DIR, 'hacks_tags_results_test.csv')
    # if os.path.isfile(deep_analysis_csv_path):
    #     df = pd.read_csv(deep_analysis_csv_path)
    # else:
    df = pd.DataFrame(columns=['file_name', 'hack title', 'brief summary', 'deep analysis_free', 'complexity', 'tags'])
    
    counter = 0

    for index, row in description_df.iterrows():
        file_name = row['file_name']     
        title = row['hack title']        
        brief_summary = row['brief summary'] 
        description = row['deep analysis_free'] 

        result_complexity, result_categories, prompt_complexity, prompt_categories  = get_hack_classifications(description)

        result_complexity = json.loads(result_complexity)
        complexity_class = result_complexity['complexity']['classification']

        result_categories = json.loads(result_categories)
        categories = [item["category"] for item in result_categories]
        tags = ", ".join(categories)

        new_row_index = len(df)
        df.loc[new_row_index] = [file_name, title, brief_summary, description, complexity_class, tags]
        counter += 1

        if counter % 10 == 0:
            df.to_csv(hacks_tagging_csv_path, index=False)
            print(f'Saved {counter} files to CSV.')
    # Save any remaining data
    if not df.empty:
        df.to_csv(hacks_tagging_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')

def split_search_results():
    queries_df = pd.read_csv(os.path.join(DATA_DIR, 'validation_queries_test.csv'))
    search_results_df = pd.read_csv(os.path.join(DATA_DIR, 'validation', 'scraping_results_f5.csv'))
    for index, row in queries_df.iterrows():
        values_to_match = json.loads(row['queries'].replace("'", '"')) 
        # print(search_results_df['query'][0] in (values_to_match))
        # subset_df = search_results_df[search_results_df['query'].isin(values_to_match)]
        subset_df = search_results_df[search_results_df.iloc[:, 0].isin(values_to_match)]
        
        # Do something with the matching results (e.g., print, save to file, etc.)
        # print(f"Matching results for query {subset_df}")
        if not subset_df.empty:
            # Save the subset as a CSV with the filename from the reference CSV
            row['file_name'] 
            subset_df.to_csv(os.path.join(DATA_DIR, 'validation', f"sources_for_validation_{row['file_name']}.csv"), index=False) 

def add_link_to_csv(csv_path= os.path.join(DATA_DIR, 'hacks_discrimination_tests_1.csv')):
    repo_base_url = "https://github.com/LignumBlocks/Hintsly/tree/main/data/test_cases/"
    df = pd.read_csv(csv_path)
    # link = f"{repo_base_url}{file_name}"
    df.insert(1, 'Link', repo_base_url + (df.iloc[:, 0].astype(str) + '.txt'))
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # df = pd.read_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv')) 
    # sorted_df = df.sort_values(by=df.columns[0])
    # sorted_df.to_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv'), index=False) 
    # process_transcriptions()
    # get_queries(os.path.join(DATA_DIR, 'hacks_discrimination_tests.csv'))
    # validate_hacks_from_csv(os.path.join(DATA_DIR, 'validation_queries_test.csv'),
    #                [os.path.join(DATA_DIR, 'validation','sources_for_validation_@hermoneymastery_video_7286913008788426027.csv'),
    #                 os.path.join(DATA_DIR, 'validation','sources_for_validation_@hermoneymastery_video_7286913008788426027.csv'),
    #                 os.path.join(DATA_DIR, 'validation','sources_for_validation_@hermoneymastery_video_7287292622924893486.csv'),
    #                 os.path.join(DATA_DIR, 'validation','sources_for_validation_@hermoneymastery_video_7301700833052314922.csv'),
    #                 os.path.join(DATA_DIR, 'validation','sources_for_validation_@hermoneymastery_video_7329918298571820331.csv')])
    # validate_hacks(os.path.join(DATA_DIR, 'validation_queries_test.csv'))
    # split_search_results()
    # analyze_validated_hacks_from_csv(os.path.join(DATA_DIR, 'validation_result_test1.csv'))
    tag_hacks_from_csv(os.path.join(DATA_DIR, 'deep_analysis_results_test.csv'))