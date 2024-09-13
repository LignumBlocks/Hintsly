import os
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
import uuid
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

models = ["gpt-4o-mini", "gpt-3.5-turbo", "gemini-1.5-flash"]
class LLMmodel:      
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        # max_tokens: int = 32000,
        # cpu_threads: int = 20,
        # gpu_layers: int = 12,
        # n_cpu_layers: int = 2,
        # batch: int = 4096,
        # CPU_only: bool=False
    ):
        self.model_name = model_name if model_name else models[0]
        self.temperature = temperature  # The temperature use for sampling, higher temperature will result in more creative and imaginative text, while lower temperature will result in more accurate and factual text.
        if 'gpt' in model_name:
            self.llm = ChatOpenAI(model=self.model_name)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                            # temperature=0.7,
                                            # max_tokens=None,
                                            # timeout=None,
                                            # max_retries=2,
                                            # other params...
                                            )
        self.session_id = str(uuid.uuid4())
        self.store = {}
    def run(self, input:str, system_prompt=None):
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder(variable_name="messages"),])
        chain = chat_prompt | self.llm
        response = chain.invoke({"messages": [HumanMessage(content=input)]})
        print("Chat response :", response.content)
        return response.content

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def run_with_history(self, input:str, system_prompt=None):
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder(variable_name="messages"),])
        chain = chat_prompt | self.llm
        self.llm_w_history = RunnableWithMessageHistory(chain, self.get_session_history)
        config = {"configurable": {"session_id": self.session_id}}
        response = self.llm_w_history.invoke([HumanMessage(content=input)], config=config)
        print("History chat response :", response.content)
        return response.content

    def simple_rag(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
