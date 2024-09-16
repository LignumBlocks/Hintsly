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
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_DISCRIMINATION'])
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

def process_transcriptions():
    # data_folder = os.path.join(DATA_DIR, 'Transcriptions Nobudgetbabe')
    hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination_10.csv')
    data_folder = os.path.join(DATA_DIR, '10_test_cases')
    # hacks_discrimination_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination_tests.csv')
    
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
        hacks_discrimination.loc[new_row_index] = [file_name_without_extension, 'tiktok Nobudgetbabe', result['is_a_hack'], result['possible hack title'], result['brief summary'], result['justification']]
        file_counter += 1
        
        # Save to CSV every 10 files
        if file_counter % 10 == 0:
            hacks_discrimination.to_csv(hacks_discrimination_csv_path, index=False)
            print(f'Saved {file_counter} files to CSV.')
    
    # Save any remaining data
    if not hacks_discrimination.empty:
        hacks_discrimination.to_csv(hacks_discrimination_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')
   
if __name__ == "__main__":
    # df = pd.read_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv')) 
    # sorted_df = df.sort_values(by=df.columns[0])
    # sorted_df.to_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv'), index=False) 
    process_transcriptions()
    # model.run("What is your favorite color?")
    # model.run_with_history("Hello, I'm Niley")
    # model.run_with_history("What is my name?")
