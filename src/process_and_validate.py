import json
import os
import llm_models
from settings import PROMPTS_TEMPLATES, DATA_DIR

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


# class HackDiscrimination(BaseModel):
#     justification: str = Field(description="Explanation about whether the content meets the criteria of a financial hack")
#     is_a_hack: bool = Field(description="Whether the content include a valid financial hack")


def verify_hacks_from_text(source_text: str):
    """ Determine whether the text constitutes a hack or not, returning structured JSON style.
    
    Args:
        source_text (str): Text content to analyse.

    Returns:
        `dict: { 
                "justification": "<analisys of whether is a hack or not>",
                "is_a_hack": "<true or false>" 
            }`
    """
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_VERIFICATION2'])
    prompt = prompt_template.format(source_text=source_text)
    system_prompt = "You are an AI financial analyst tasked with classifying content related to financial strategies."
    # print(prompt)
    # return
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result:str = model.run(prompt, system_prompt)
        cleaned_string = result.replace("```json\n", "").replace("```","")
        # Strip leading and trailing whitespace
        cleaned_string = cleaned_string.strip()
        return json.loads(cleaned_string), prompt
    except Exception as er:
        print(f"Error in verify hacks: {er}")
        return None, prompt
    
def get_queries_for_validation(hack_title: str, source_text: str, num_queries: int=4):
    """ For a hack summary select queries to validate against real sources.
    
    Args:
        source_text (str): Text content to analyse.

    Returns:
        `list: relevant queries for the given text`
    """
    
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['GET_QUERIES'])
    prompt = prompt_template.format(hack_title=hack_title, hack_summary=source_text, num_queries=num_queries)
    system_prompt = "You are an AI financial analyst tasked with accepting or refusing the validity of a financial hack."
    
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result:str = model.run(prompt, system_prompt)
        cleaned_string = result.replace("```json\n", "").replace("```","")
        # Strip leading and trailing whitespace
        cleaned_string = cleaned_string.strip()
        return json.loads(cleaned_string), prompt
    except Exception as er:
        print(f"Error getting the queries for the hacks: {er}")
        return None, prompt

def validate_financial_hack(hack_source, hack_title: str, hack_summary: str, query_df: str):
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        rag = llm_models.RAG_LLMmodel("gpt-4o-mini",chroma_path=os.path.join(DATA_DIR, 'chroma_db'))
        rag.store_from_query_csv(query_df, hack_source)
        chunks = ""
        metadata = []
        # print(model.vector_store)
        for result in rag.retrieve_similar_for_hack(hack_source, hack_title+ ':\n'+hack_summary):
            print(result.metadata)
            metadata.append((result.metadata['link'], result.metadata['source']))
            chunks += result.page_content + "\n"

        prompt_template:str = load_prompt(PROMPTS_TEMPLATES['VALIDATE_HACK'])
        prompt = prompt_template.format(chunks=chunks, hack_title=hack_title, hack_summary=hack_summary)
        system_prompt = "You are an AI financial analyst tasked with accepting or refusing the validity of a financial hack."
        
        result:str = model.run(prompt, system_prompt)
        try:
            cleaned_string = result.replace("```json\n", "").replace("```","")
            # Strip leading and trailing whitespace
            cleaned_string = cleaned_string.strip()
            result = cleaned_string
        except:
            pass
        return json.loads(result), prompt, metadata
    except Exception as er:
        print(f"Error validating hacks: {er}")
        return None, None, None

def get_deep_analysis(hack_title: str, hack_summary: str, original_text: str):
    prompt_template_free:str = load_prompt(PROMPTS_TEMPLATES['DEEP_ANALYSIS_FREE'])
    prompt_template_premium:str = load_prompt(PROMPTS_TEMPLATES['DEEP_ANALYSIS_PREMIUM'])
    prompt_free = prompt_template_free.format(hack_title=hack_title, hack_summary=hack_summary, original_text=original_text)
    system_prompt = "You are a financial analyst specializing in creating financial hacks for users in the USA"
    
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result_free = model.run(prompt_free, system_prompt)
        prompt_premium = prompt_template_premium.format(hack_title=hack_title, hack_summary=hack_summary, original_text=original_text,free_analysis=result_free)
        result_premium = model.run(prompt_premium, system_prompt)
        return result_free, result_premium, prompt_free, prompt_premium 
    except Exception as er:
        print(f"Error in deep_analysis: {er}")
        return None, None, prompt_free, prompt_premium

def get_structured_analysis(result_free: str, result_premium: str):
    prompt_template_free:str = load_prompt(PROMPTS_TEMPLATES['STRCT_DEEP_ANALYSIS_FREE'])
    prompt_template_premium:str = load_prompt(PROMPTS_TEMPLATES['STRCT_DEEP_ANALYSIS_PREMIUM'])
    prompt_free = prompt_template_free.format(free_analysis=result_free)
    prompt_premium = prompt_template_premium.format(premium_analysis=result_premium)
    system_prompt = "You are a financial analyst specializing in creating financial hacks for users in the USA"
    
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result_free = model.run(prompt_free, system_prompt)
        result_premium = model.run(prompt_premium, system_prompt)
        try:
            cleaned_string = result_free.replace("```json\n", "").replace("```","")
            # Strip leading and trailing whitespace
            cleaned_string = cleaned_string.strip()
            result_free = cleaned_string
        except:
            pass
        try:
            cleaned_string = result_premium.replace("```json\n", "").replace("```","")
            # Strip leading and trailing whitespace
            cleaned_string = cleaned_string.strip()
            result_premium = cleaned_string
        except:
            pass
        return json.loads(result_free), json.loads(result_premium), prompt_free, prompt_premium 
    except Exception as er:
        print(f"Error in deep_analysis: {er}")
        return None, None, prompt_free, prompt_premium

def get_hack_classifications(result_free: str):
    prompt_template_complexity:str = load_prompt(PROMPTS_TEMPLATES['COMPLEXITY_TAG'])
    prompt_template_categories:str = load_prompt(PROMPTS_TEMPLATES['CLASIFICATION_TAGS'])
    prompt_complexity = prompt_template_complexity.format(hack_description=result_free)
    prompt_categories = prompt_template_categories.format(hack_description=result_free)
    system_prompt = "You are a financial analyst specializing in creating financial hacks for users in the USA"
    
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result_complexity = model.run(prompt_complexity, system_prompt)
        result_categories = model.run(prompt_categories, system_prompt)
        try:
            cleaned_string = result_complexity.replace("```json\n", "").replace("```","")
            # Strip leading and trailing whitespace
            cleaned_string = cleaned_string.strip()
            result_complexity = cleaned_string
        except:
            pass
        try:
            cleaned_string = result_categories.replace("```json\n", "").replace("```","")
            # Strip leading and trailing whitespace
            cleaned_string = cleaned_string.strip()
            result_categories = cleaned_string
        except:
            pass
        return json.loads(result_complexity), json.loads(result_categories), prompt_complexity, prompt_categories 
    except Exception as er:
        print(f"Error in deep_analysis: {er}")
        return None, None, prompt_complexity, prompt_categories

