import os
import json
import pandas as pd
import load_model
from settings import BASE_DIR, SRC_DIR, DATA_DIR, PROMPT_DIR, PROMPTS_TEMPLATES
from langchain_core.pydantic_v1 import BaseModel, Field

def load_prompt(*args):
    """Constructs a prompt from the prompting files in the prompts directory.

    Args:
        args (str): The names of the prompting files to include in the prompt.

    Returns:
        str: The constructed prompt."""

    prompt = ""
    for file_path in args:
        with open(file_path, "r") as file:
            prompt += file.read().strip()
    return prompt

class HackDiscrimination(BaseModel):
    justification: str = Field(description="Explanation about whether the content meets the criteria of a financial hack")
    is_a_hack: bool = Field(description="Whether the content include a valid financial hack")

def discriminate_hacks_from_text(source_text: str, source: str):
    """ Determine whether the text constitutes a hack or not, returning structured JSON style.
    
    Args:
        source_text (str): Text content to analyse.

    Returns:
        `dict: { 
                "justification": "<analisys of whether is a hack or not>",
                "is_a_hack": "<true or false>" 
            }`
    """
    # prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_DISCRIMINATION0'])
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_DISCRIMINATION1'])
    # prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_DISCRIMINATION2'])
    prompt = prompt_template.format(source_text=source_text, source=source)
    system_prompt = "You are an AI financial analyst tasked with classifying content related to financial strategies."
    # print(prompt)
    # return
    try:
        model = load_model.LLMmodel("gpt-4o-mini")
        result:str = model.run(prompt, system_prompt)
        cleaned_string = result.replace("```json\n", "").replace("```","")
        # Strip leading and trailing whitespace
        cleaned_string = cleaned_string.strip()
        return json.loads(cleaned_string), prompt
    except Exception as er:
        print(f"Error discriminating hacks: {er}")
        return None, prompt
    
def get_queries_for_validation(source_text: str, num_queries: int=4):
    """ For a hack summary select validate against real sources.
    
    Args:
        source_text (str): Text content to analyse.

    Returns:
        `list: relevant queries for the given text`
    """
    
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['GET_QUERIES'])
    prompt = prompt_template.format(hack_summary=source_text, num_queries=num_queries)
    system_prompt = "You are an AI financial analyst tasked with accepting or refusing the validity of a financial hack."
    
    try:
        model = load_model.LLMmodel("gpt-4o-mini")
        result:str = model.run(prompt, system_prompt)
        cleaned_string = result.replace("```json\n", "").replace("```","")
        # Strip leading and trailing whitespace
        cleaned_string = cleaned_string.strip()
        return json.loads(cleaned_string), prompt
    except Exception as er:
        print(f"Error discriminating hacks: {er}")
        return None, prompt

def get_queries(csv_path: str):
    source_df = pd.read_csv(csv_path)
    validation_queries_csv_path = os.path.join(DATA_DIR, 'validation_queries_test1.csv')
    if os.path.isfile(validation_queries_csv_path):
        df = pd.read_csv(validation_queries_csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'brief summary', 'queries'])
    
    counter = 0

    for index, row in source_df.iterrows():
        file_name = row['file_name']        
        brief_summary = row['brief summary'] 

        query_results, prompt = get_queries_for_validation(brief_summary)
        query_results = query_results['queries']
        new_row_index = len(df)
        df.loc[new_row_index] = [file_name, brief_summary, query_results]
        counter += 1

        if counter % 10 == 0:
            df.to_csv(validation_queries_csv_path, index=False)
            print(f'Saved {counter} files to CSV.')
    # Save any remaining data
    if not df.empty:
        df.to_csv(validation_queries_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')

def process_transcriptions():
    # data_folder = os.path.join(DATA_DIR, 'Transcriptions Nobudgetbabe')
    # hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination.csv')
    data_folder = os.path.join(DATA_DIR, 'test_cases')
    hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination_tests_1.csv')
    
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
    get_queries(os.path.join(DATA_DIR, 'hacks_discrimination_tests_0.csv'))
    # model.run("What is your favorite color?")
    # model.run_with_history("Hello, I'm Niley")
    # model.run_with_history("What is my name?")
