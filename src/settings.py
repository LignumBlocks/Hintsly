import os
# paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # Hintsly/src
BASE_DIR = os.path.dirname(SRC_DIR)                    # Hintsly
TEST_DIR = os.path.join(BASE_DIR, 'test')              # Hintsly/test
DATA_DIR = os.path.join(BASE_DIR, 'data')              # Hintsly/data
PROMPT_DIR = os.path.join(SRC_DIR, 'prompts') 

# prompts
PROMPTS_TEMPLATES = {
    'HACK_DISCRIMINATION0':os.path.join(PROMPT_DIR, "ishack", "hack_discrimination"),
    'HACK_DISCRIMINATION1':os.path.join(PROMPT_DIR, "ishack", "hack_discrimination_medium"),
    'HACK_DISCRIMINATION2':os.path.join(PROMPT_DIR, "ishack", "hack_discrimination_reduced"),
    'GET_QUERIES':os.path.join(PROMPT_DIR, "validation", "generate_questions"),
    'VALIDATE_HACK':os.path.join(PROMPT_DIR, "validation", "rag_to_evaluate"),
    }