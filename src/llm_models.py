import os
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
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
from datetime import datetime, timezone
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
   

class RAG_LLMmodel:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.4, chroma_path='chroma_db',collection_name='validation'):
        self.model_name = model_name if model_name else models[0]
        self.temperature = temperature  # The temperature use for sampling, higher temperature will result in more creative and imaginative text, while lower temperature will result in more accurate and factual text.
        if 'gpt' in model_name:
            self.llm = ChatOpenAI(model=self.model_name)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                              cache=False, temperature=temperature)
        self.chroma_path = chroma_path
        self.embeddings = OpenAIEmbeddings() # TODO generalize like self.llm
        self.vectorstore = self.load_or_create_chroma(collection_name)
        # self.vectorstore.persist()
        # https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
        # prompt = hub.pull("rlm/rag-prompt")
        # self.qa_chain = RetrievalQA.from_llm(
        #     self.llm, retriever=retriever, prompt=prompt
        # )
        # response = self.qa_chain.run(text_to_compare)

    def load_or_create_chroma(self, collection_name):
        """Loads existing Chroma store or creates a new one if it doesn't exist."""
        try:
            return Chroma(collection_name=collection_name,persist_directory=self.chroma_path, embedding_function=self.embeddings)
        except FileNotFoundError:
            return Chroma.from_documents([], self.embeddings, persist_directory=self.chroma_path)

    def add_document(self, hack_id, document_title, documents, metadata):
        """Adds a new document to the Chroma vector store only if it doesn't exist."""
        # Check if document already exists for the hack
        existing_documents = self.vectorstore.get()
            # where={"hack_id": hack_id, "title": document_title})
        filtered_documents = [doc for doc in existing_documents['metadatas'] if (doc["hack_id"] == hack_id and doc["title"] == document_title)]

        # print(f'existing_documents for {hack_id}', len(filtered_documents), documents[0])
        # Add to Chroma vector store
        if len(filtered_documents) == 0:
            self.vectorstore.add_texts(documents, metadatas=metadata)
        # self.vectorstore.persist()

    def query_for_hack(self, hack_id, text_to_compare):
        retriever = self.vectorstore.as_retriever(
            filter={"hack_id": hack_id}
        )
        prompt = hub.pull("rlm/rag-prompt")
        self.qa_chain = RetrievalQA.from_llm(
            self.llm, retriever=retriever, prompt=prompt
        )
        response = self.qa_chain.run(text_to_compare)

    def retrieve_similar_for_hack(self, hack_id, text_to_compare, k=4):
        """Retrieves the top k most similar documents to the given text."""
        similar_documents = self.vectorstore.similarity_search(query=text_to_compare,k=k,filter={"hack_id": hack_id})
        # retriever = self.vectorstore.as_retriever(
        #     filter={"hack_id": hack_id}
        # )
        # similar_documents = retriever.get_relevant_documents(text_to_compare, k=k)
        return similar_documents

    def simple_rag(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        return vectorstore

    def retrieve_similar_chunks(self, query):
        results = self.vector_store.similarity_search(query, k=4)
        return results
    
    def store_from_query_csv(self, query_df: str, hack_source):
        
        # if os.path.isdir(persist_directory):
        #     print('loading from persist directory')
        #     self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        #     print('vectore_store ready')
        #     return self.vector_store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
        
        for index, row in query_df.iterrows():
            if not pd.isna(row['content']) and row['content'] != '' and row['content'] != 'Error al cargar el contenido':
                documents = []
                metadatas = []
                content_chunks = text_splitter.split_text(row['content'])
                for chunk in content_chunks:
                    documents.append(chunk)
                    metadatas.append({
                        "hack_id": hack_source,
                        "query": row['query'],
                        "source": row['source'],
                        "title": row['title'],
                        "description": row['description'],
                        "link": row['link']
                    })
                self.add_document(hack_source, row['title'], documents, metadatas)
                    # print(documents[-1])
                    # print(metadatas[-1])
        # data = { "documents": documents, "metadatas": metadatas }

        # with open('data.json', 'w') as json_file:
        #     json.dump(data, json_file)

