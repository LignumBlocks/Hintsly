import os
import load_model
from settings import BASE_DIR, SRC_DIR

if __name__ == "__main__":
    model = load_model.LLMmodel()
    model.run("What is your favorite color?")
    model.run_with_history("What is my name?")
