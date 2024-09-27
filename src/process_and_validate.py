import json
import os
import llm_models
from settings import PROMPTS_TEMPLATES, DATA_DIR

def load_prompt(*args):
    """
    Constructs a prompt by loading the content from one or more prompt template files in the prompts directory.

    Args:
        args (str): The file paths of the prompt templates to load.

    Returns:
        str: The combined content of the loaded prompt templates.
    """

    prompt = ""
    for file_path in args:
        with open(file_path, "r") as file:
            prompt += file.read().strip()
    return prompt

def verify_hacks_from_text(source_text: str):
    """
    Analyzes a piece of text to determine if it constitutes a financial hack, returning a JSON result.

    Args:
        source_text (str): The text content to analyze.

    Returns:
        dict: A dictionary with justification and a boolean indicating whether the text is considered a hack.
        str: The constructed prompt used to generate the result.
    """
    prompt_template:str = load_prompt(PROMPTS_TEMPLATES['HACK_VERIFICATION2'])
    prompt = prompt_template.format(source_text=source_text)
    system_prompt = "You are an AI financial analyst tasked with classifying content related to financial strategies."
    
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
    """
    Generates a list of queries to validate a financial hack against real-world sources.

    Args:
        hack_title (str): The title of the hack.
        source_text (str): The summary of the hack.
        num_queries (int): The number of queries to generate (default is 4).

    Returns:
        list: A list of relevant queries for validating the hack.
        str: The constructed prompt used to generate the result.
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

def validate_financial_hack(hack_id, hack_title: str, hack_summary: str, queries_dict: list):
    """
    Validates a financial hack by retrieving relevant documents from a vector store and analyzing the information.

    Args:
        hack_id (str): The identifier of the hack.
        hack_title (str): The title of the hack.
        hack_summary (str): The summary of the hack.
        queries_dict (list): A list of queries for validating the hack.

    Returns:
        dict: A dictionary with the validation results.
        str: The constructed prompt used for validation.
        list: Metadata of the retrieved documents.
    """
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        rag = llm_models.RAG_LLMmodel("gpt-4o-mini", chroma_path=os.path.join(DATA_DIR, 'chroma_db'))
        rag.store_from_queries(queries_dict, hack_id)
        chunks = ""
        metadata = []
        for result in rag.retrieve_similar_for_hack(hack_id, hack_title+ ':\n'+hack_summary):
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
    """
    Performs a deep analysis of a financial hack by generating both free and premium-level analysis.

    Args:
        hack_title (str): The title of the hack.
        hack_summary (str): The summary of the hack.
        original_text (str): The original source text for the hack.

    Returns:
        tuple: A tuple containing free and premium-level analysis results along with the prompts used.
    """
    prompt_template_free:str = load_prompt(PROMPTS_TEMPLATES['DEEP_ANALYSIS_FREE'])
    prompt_template_premium:str = load_prompt(PROMPTS_TEMPLATES['DEEP_ANALYSIS_PREMIUM'])
    prompt_free = prompt_template_free.format(hack_title=hack_title, hack_summary=hack_summary, original_text=original_text)
    system_prompt = "You are a financial analyst specializing in creating financial hacks for users in the USA."
    
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")
        result_free = model.run(prompt_free, system_prompt)
        prompt_premium = prompt_template_premium.format(hack_title=hack_title, hack_summary=hack_summary, original_text=original_text,free_analysis=result_free)
        result_premium = model.run(prompt_premium, system_prompt)
        return result_free, result_premium, prompt_free, prompt_premium 
    except Exception as er:
        print(f"Error in deep_analysis: {er}")
        return None, None, None, None

def enriched_analysis(free_description, premium_description, chunks):
    """
    Performs an enriched analysis by further refining both free and premium analyses of a financial hack.

    Args:
        free_description (str): The result of the free analysis.
        premium_description (str): The result of the premium analysis.
        chunks (str): Additional text or documents related to the hack.

    Returns:
        tuple: A tuple containing updated free and premium analyses along with the prompts used.
    """
    try:
        model = llm_models.LLMmodel("gpt-4o-mini")

        prompt_template_free:str = load_prompt(PROMPTS_TEMPLATES['ENRICHED_ANALYSIS_FREE'])
        prompt_template_premium:str = load_prompt(PROMPTS_TEMPLATES['ENRICHED_ANALYSIS_PREMIUM'])
        free_prompt = prompt_template_free.format(chunks=chunks, previous_analysis=free_description)
        system_prompt = "You are a financial analyst specializing in creating financial hacks for users in the USA."
        
        result_free = model.run(free_prompt, system_prompt)
        premium_prompt = prompt_template_premium.format(chunks=chunks, free_analysis=result_free, previous_analysis=premium_description)
        result_premium = model.run(premium_prompt, system_prompt)
        
        return result_free, result_premium, free_prompt, premium_prompt 
    except Exception as er:
        print(f"Error deepening hacks descriptions: {er}")
        return None, None, None

def get_structured_analysis(result_free: str, result_premium: str):
    """
    Extract the structured information from the free hack description and the premium hack description.

    Args:
        result_free (str): The result of the free hack description.
        result_premium (str): The result of the premium hack description.

    Returns:
        tuple: A tuple containing structured analysis results (in JSON format) for both free and premium analyses along with the prompts used.
    """
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
    """
    Classifies a financial hack based on several parameters.

    Args:
        result_free (str): The result of the enriched free hack description.

    Returns:
        tuple: A tuple containing the classification for each parameter.
    """
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
