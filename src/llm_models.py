import os
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain import hub
import uuid
from dotenv import load_dotenv
from settings import DATA_DIR

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

models = ["gpt-4o-mini", "gpt-3.5-turbo", "gemini-1.5-flash"]
class LLMmodel:      
    """
    A class to manage interactions with different LLMs like OpenAI's GPT or Google's Gemini models.
    The class supports managing conversations with or without message history.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initializes the LLMmodel with the specified model and temperature.
        
        Args:
            model_name (str): The name of the LLM model to use (default is "gpt-4o-mini").
            temperature (float): Controls the randomness of model outputs (default is 0.7).
        """
        self.model_name = model_name if model_name else models[0]
        self.temperature = temperature
        # Select the appropriate LLM model
        if 'gpt' in model_name:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                              cache=False,
                                              temperature=self.temperature)
        
        # Generate a session ID for tracking
        self.session_id = str(uuid.uuid4())
        self.store = {}
        self.trimmer = trim_messages(max_tokens=8000,
                                    strategy="last",
                                    token_counter=self.llm,
                                    include_system=True,
                                    allow_partial=False,
                                    start_on="human",)
    
    def run(self, input: str, system_prompt=None):
        """
        Executes a conversation with the LLM based on the provided input and optional system prompt.
        
        Args:
            input (str): User input or query.
            system_prompt (str): An optional system message to guide the LLM's response (default is a generic helper).
        
        Returns:
            str: The LLM's response content.
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                        MessagesPlaceholder(variable_name="messages"),])
        # Set up the conversation chain
        chain = (RunnablePassthrough.assign(messages=itemgetter("messages") | self.trimmer) 
                 | chat_prompt | self.llm)
        response = chain.invoke({"messages": [HumanMessage(content=input)]})
        print("Chat response :", response.content)
        return response.content

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves or creates an in-memory chat message history for a session.
        
        Args:
            session_id (str): The session ID for the conversation.
        
        Returns:
            BaseChatMessageHistory: The chat history object for the session.
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def run_with_history(self, input: str, system_prompt=None):
        """
        Executes a conversation with the LLM while tracking the conversation history.
        
        Args:
            input (str): User input or query.
            system_prompt (str): An optional system message to guide the LLM's response.
        
        Returns:
            str: The LLM's response content.
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Answer all questions to the best of your ability."
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                        MessagesPlaceholder(variable_name="messages"),])
        chain = chat_prompt | self.llm
        # Combine LLM with message history tracking
        self.llm_w_history = RunnableWithMessageHistory(chain, self.get_session_history)
        config = {"configurable": {"session_id": self.session_id}}
        response = self.llm_w_history.invoke([HumanMessage(content=input)], config=config)
        print("History chat response :", response.content)
        return response.content

class RAG_LLMmodel:
    """
    A class for managing a Retrieval-Augmented Generation (RAG) model using LLMs and a Chroma vector store.
    This class allows the addition and retrieval of documents from a vector store for better context-aware conversations.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.4, chroma_path='chroma_db', collection_name='validation'):
        """
        Initializes the RAG_LLMmodel with the specified model, temperature, Chroma path, and collection name.
        
        Args:
            model_name (str): The name of the LLM model to use (default is "gpt-4o-mini").
            temperature (float): Controls the randomness of model outputs (default is 0.4).
            chroma_path (str): The path to the Chroma database (default is 'chroma_db').
            collection_name (str): The name of the collection in Chroma (default is 'validation').
        """
        self.model_name = model_name if model_name else models[0]
        self.temperature = temperature
        if 'gpt' in model_name:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                              cache=False, temperature=self.temperature)
        self.chroma_path = chroma_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # TODO generalize for gemini like self.llm
        self.vectorstore = self.load_or_create_chroma(collection_name)
       
    def load_or_create_chroma(self, collection_name):
        """
        Loads an existing Chroma vector store or creates a new one if it doesn't exist.
        
        Args:
            collection_name (str): The name of the Chroma collection.
        
        Returns:
            Chroma: The Chroma vector store object.
        """
        # https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
        try:
            return Chroma(collection_name=collection_name,persist_directory=self.chroma_path, embedding_function=self.embeddings)
        except FileNotFoundError:
            return Chroma.from_documents([], self.embeddings, persist_directory=self.chroma_path)

    def simple_rag(self, docs):
        """
        Performs a simple Retrieval-Augmented Generation process by splitting documents into chunks and storing them in a vector store.
        
        Args:
            docs (list): A list of documents to split and store.
        
        Returns:
            Chroma: The updated vector store containing the document chunks.
        """
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splits, self.embeddings)
        return vectorstore
    
    def add_document(self, hack_id, document_title, documents, metadata):
        """
        Adds a new document to the Chroma vector store if it doesn't already exist.
        
        Args:
            hack_id (str): The identifier for the document.
            document_title (str): The title of the document.
            documents (list): A list of document content.
            metadata (dict): Metadata to associate with the document in the vector store.
        """
        existing_documents = self.vectorstore.get()
        
        filtered_documents = [doc for doc in existing_documents['metadatas'] if (doc["hack_id"] == hack_id and doc["title"] == document_title)]

        # print(f'existing_documents for {hack_id}', len(filtered_documents), documents[0])
        if len(filtered_documents) == 0:
            self.vectorstore.add_texts(documents, metadatas=metadata)

    def retrieve_similar_for_hack(self, hack_id, text_to_compare, k=4):
        """
        Retrieves the top-k most similar documents to the given text for a specific hack ID.
        
        Args:
            hack_id (str): The identifier for the document.
            text_to_compare (str): The text to compare for similarity search.
            k (int): The number of similar documents to retrieve (default is 4).
        
        Returns:
            list: A list of similar documents.
        """
        similar_documents = self.vectorstore.similarity_search(query=text_to_compare,k=k,filter={"hack_id": hack_id})
        return similar_documents

    def store_from_queries(self, queries_dict: list, hack_id):
        """
        Stores documents from a list of query results into the Chroma vector store, splitting them into chunks.
        
        Args:
            queries_dict (list): A list of dictionaries containing query results.
            hack_id (str): The identifier for the documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        
        for query_dict in queries_dict:
            if query_dict['content'] != '' and query_dict['content'] != 'Error al cargar el contenido':
                documents = []
                metadatas = []
                content_chunks = text_splitter.split_text(query_dict['content'])
                for chunk in content_chunks:
                    documents.append(chunk)
                    metadatas.append({
                        "hack_id": hack_id,
                        "query": query_dict['query'],
                        "source": query_dict['source'],
                        "title": query_dict['title'],
                        "description": query_dict['description'],
                        "link": query_dict['link']
                    })
                self.add_document(hack_id, query_dict['title'], documents, metadatas)
