import os
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain import hub
import uuid
import pandas as pd
import json
from dotenv import load_dotenv
from settings import DATA_DIR

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

models = ["gpt-4o-mini", "gpt-3.5-turbo", "gemini-1.5-flash"]
class LLMmodel:      
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        # output_structure=None
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
                                              cache=False,
                                              temperature=0.4,
                                            # max_tokens=None,
                                            # timeout=None,
                                            # max_retries=2,
                                            # other params...
                                            )
        
        self.session_id = str(uuid.uuid4())
        self.store = {}
        self.trimmer = trim_messages(max_tokens=8000,
                                    strategy="last",
                                    token_counter=self.llm,
                                    include_system=True,
                                    allow_partial=False,
                                    start_on="human",)
    
    def classify_run(self, input: str, system_prompt: str, output_structure):
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        tagging_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder(variable_name="messages")])
        self.llm_structured = self.llm.with_structured_output(output_structure)
        tagging_chain = tagging_prompt | self.llm_structured
        
        response = tagging_chain.invoke({"messages": [HumanMessage(content=input)]})
        print(response)
        return response
    
    def run(self, input: str, system_prompt=None):
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                        MessagesPlaceholder(variable_name="messages"),])
        chain = (RunnablePassthrough.assign(messages=itemgetter("messages") | self.trimmer) 
                 | chat_prompt | self.llm)
        response = chain.invoke({"messages": [HumanMessage(content=input)]})
        print("Chat response :", response.content)
        return response.content

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def run_with_history(self, input: str, system_prompt=None):
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                        MessagesPlaceholder(variable_name="messages"),])
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

    def retrieve_similar_chunks(self, query):
        results = self.vector_store.similarity_search(query, k=4)
        return results
    
    def vector_store_from_query_csv(self, csv_path: str):
        persist_directory = os.path.join(DATA_DIR, "chroma_langchain_db")
        embeddings = OpenAIEmbeddings()
        # if os.path.isdir(persist_directory):
        #     print('loading from persist directory')
        #     self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        #     print('vectore_store ready')
        #     return self.vector_store
        df = pd.read_csv(csv_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        documents = []
        metadatas = []
        for index, row in df.iterrows():
            if not pd.isna(row['content']) and row['content'] != '' and row['content'] != 'Error al cargar el contenido':
                content_chunks = text_splitter.split_text(row['content'])
                for chunk in content_chunks:
                    documents.append(chunk)
                    metadatas.append({
                        "query": row['query'],
                        "source": row['source'],
                        "title": row['title'],
                        "description": row['description'],
                        "link": row['link']
                    })
                    # print(documents[-1])
                    # print(metadatas[-1])
        data = { "documents": documents, "metadatas": metadatas }

        with open('data.json', 'w') as json_file:
            json.dump(data, json_file)

        self.vector_store = Chroma.from_texts(documents, embeddings, metadatas=metadatas, persist_directory=persist_directory,
                                            collection_name="hermoneymastery_video_7254212698383142187")
        
        print('vectore_store ready')
        return self.vector_store
