import os
import pandas as pd
import load_model
from settings import BASE_DIR, SRC_DIR, DATA_DIR, PROMPTS_TEMPLATES
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

def discriminate_hacks_from_text(model: load_model.LLMmodel, source_text: str):
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
    prompt = prompt_template.format(source_text=source_text)
    system_prompt = "You are an AI financial analyst tasked with classifying content related to financial strategies."
    result = model.classify_run(prompt, system_prompt, HackDiscrimination)
    return result.dict()

def process_transcriptions():
    data_folder = os.path.join(DATA_DIR, 'Transcriptions Nobudgetbabe')
    output_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination.csv')
    hacks_discrimination = pd.DataFrame(columns=['file_name', 'source', 'hack_status', 'justification'])
    
    model = load_model.LLMmodel()
    file_counter = 0
    
    # Iterate over each file in the directory
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_folder, file_name)
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Process the text content
            result = discriminate_hacks_from_text(model, text_content)
            file_name_without_extension = os.path.splitext(file_name)[0]
            # Prepare the new row
            new_row_index = len(hacks_discrimination)  # Get the next index
            hacks_discrimination.loc[new_row_index] = [file_name_without_extension, 'tiktok Nobudgetbabe', result['is_a_hack'], result['justification']]
            file_counter += 1
            
            # Save to CSV every 10 files
            if file_counter % 10 == 0:
                hacks_discrimination.to_csv(output_csv_path, index=False)
                print(f'Saved {file_counter} files to CSV.')
    
    # Save any remaining data
    if not hacks_discrimination.empty:
        hacks_discrimination.to_csv(output_csv_path, index=False)
        print('Final save: saved remaining files to CSV.')
   
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv')) 
    sorted_df = df.sort_values(by=df.columns[0])
    sorted_df.to_csv(os.path.join(DATA_DIR, 'hacks_discrimination.csv'), index=False) 
    # process_transcriptions()
    # model.run("What is your favorite color?")
    # model.run_with_history("Hello, I'm Niley")
    # model.run_with_history("What is my name?")
