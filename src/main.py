import os
import pandas as pd
import load_model
from settings import BASE_DIR, SRC_DIR, DATA_DIR
from langchain_core.pydantic_v1 import BaseModel, Field

class Classification(BaseModel):
    justification: str = Field(description="Explanation about whether the content meets the criteria of a financial hack")
    is_a_hack: bool = Field(description="Whether the content include a valid financial hack")

def discriminate_hacks_from_text(source_text: str):
    """ Given a text input determine whether it constitute a hack or not, it can be more than 
    one hack per document. The output must be a structured json style with the fields:
    { 
      "justification": "<analisys of whether is a hack or not>",
      "is_a_hack": <true or false> 
    }
    """
    hack_definition = """"""
    instructions = """"""

def process_transcriptions():
    data_folder = os.path.join(DATA_DIR, 'Transcriptions Nobudgetbabe')
    output_csv_path = os.path.join(DATA_DIR, 'hacks_discrimination.csv')
    hacks_discrimination = pd.DataFrame(columns=['file_name', 'source', 'hack_status', 'justification'])
    
    file_counter = 0
    
    # Iterate over each file in the directory
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_folder, file_name)
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Process the text content
            result = discriminate_hacks_from_text(text_content)
            file_name_without_extension = os.path.splitext(file_name)[0]
            # Prepare the new row
            new_row = {
                'file_name': file_name_without_extension,
                'source': 'tiktok Nobudgetbabe',  # Replace with your actual source
                'hack_status': result['is_a_hack'],
                'justification': result['justification']
            }
            hacks_discrimination = hacks_discrimination.append(new_row, ignore_index=True)
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
    model = load_model.LLMmodel()
    model.run("What is your favorite color?")
    model.run_with_history("Hello, I'm Niley")
    model.run_with_history("What is my name?")
