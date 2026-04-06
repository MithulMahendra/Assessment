"""
=============================================================
  PYTHON CODING CHALLENGE
  Topic   : LangChain v1 · RAG Agents · pgvector ·
            Embeddings · LangSmith
  Level   : Intermediate
  Tasks   : 20  (project-style, grouped by topic)
=============================================================

SETUP — install dependencies before you begin
----------------------------------------------
  pip install langchain langchain-openai langchain-community
              langchain-core langsmith psycopg2-binary numpy
              python-dotenv

ENVIRONMENT VARIABLES — create a .env file or export these:
  OPENAI_API_KEY       = "sk-..."
  LANGCHAIN_API_KEY    = "ls__..."        # LangSmith
  LANGCHAIN_TRACING_V2 = "true"
  LANGCHAIN_PROJECT    = "rag-challenge"
  PG_CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/vectordb"

TOPIC SECTIONS
--------------
  Section A — LangChain Core         (Tasks  1 – 4)
  Section B — Embeddings             (Tasks  5 – 8)
  Section C — pgvector               (Tasks  9 – 13)
  Section D — RAG Agents             (Tasks 14 – 17)
  Section E — LangSmith              (Tasks 18 – 20)

RULES
-----
  - Implement every function stub below.
  - Do NOT add extra libraries beyond those listed in Setup.
  - Keep function signatures exactly as given.
  - For tasks that call an LLM, handle API errors gracefully
    with try/except.
=============================================================
"""

import os
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
import psycopg2
import json
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.tools.retriever import create_retriever_tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ─────────────────────────────────────────────────────────────
# SECTION A — LangChain Core  (Tasks 1 – 4)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 1 — Basic LCEL Chain with PromptTemplate
# ─────────────────────────────────────────────────────────────
"""
TASK 1: Basic LCEL Chain
--------------------------
Build a simple LangChain Expression Language (LCEL) chain that:
  1. Accepts a topic as input.
  2. Fills it into a PromptTemplate.
  3. Sends the prompt to ChatOpenAI (gpt-4o-mini).
  4. Parses the output as a plain string.
  5. Returns the result.

Use the pipe operator  |  to chain components.

Expected usage:
  result = basic_lcel_chain("quantum computing")
  print(result)
  # "Quantum computing uses quantum bits (qubits)..."

HINT:
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain_core.output_parsers import StrOutputParser

  chain = prompt | llm | parser
  chain.invoke({"topic": "..."})
"""

def basic_lcel_chain(topic: str) -> str:
    """Returns a one-paragraph explanation of the given topic."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    prompt = ChatPromptTemplate.from_template(
        "Please provide a one-paragraph explanation of the following topic: {topic}"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    
    result = chain.invoke({"topic": topic})
    
    return result

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 2 — Sequential Chain (Multi-Step Pipeline)
# ─────────────────────────────────────────────────────────────
"""
TASK 2: Sequential Chain (Multi-Step Pipeline)
------------------------------------------------
Build a two-step LCEL chain:
  Step 1 — given a topic, generate a short 3-sentence summary.
  Step 2 — given the summary, translate it into French.

Return a dict with keys:
  {"summary": "...", "translation": "..."}

HINT:
  - Use RunnablePassthrough or RunnableParallel to pass
    intermediate outputs to the next step.
  - from langchain_core.runnables import RunnablePassthrough
  - Chain: (prompt1 | llm | parser) then feed output
    into (prompt2 | llm | parser).
  - You can use two separate .invoke() calls if you prefer
    to keep it simple and readable.
"""

def sequential_chain(topic: str) -> dict:
    """Returns {'summary': ..., 'translation': ...} for the topic."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
        
    prompt1 = ChatPromptTemplate.from_template("Write a short 3-sentence summary about the following topic: {topic}")
    summary_chain = prompt1 | llm | parser
    prompt2 = ChatPromptTemplate.from_template("Translate the following text into French:\n\n{summary}")
    translation_chain = prompt2 | llm | parser
        
    full_chain = (
        {"summary": summary_chain} 
        | RunnablePassthrough.assign(translation=translation_chain)
    )
        
    return full_chain.invoke({"topic": topic})
        
    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 3 — Conversation Chain with Memory
# ─────────────────────────────────────────────────────────────
"""
TASK 3: Conversation Chain with Memory
----------------------------------------
Build a conversational chain that:
  - Maintains chat history across multiple turns.
  - Uses ChatPromptTemplate with a MessagesPlaceholder
    for the history.
  - Returns a list of (role, content) tuples representing
    the full conversation after all turns.

Simulate this 3-turn conversation:
  Turn 1 — user: "My name is Alex. What is machine learning?"
  Turn 2 — user: "Can you give me a real-world example?"
  Turn 3 — user: "What is my name?"   ← tests memory

Expected (partial):
  [("human", "My name is Alex..."),
   ("ai",    "Machine learning is..."),
   ...
   ("ai",    "Your name is Alex.")]

HINT:
  from langchain_core.chat_history import InMemoryChatMessageHistory
  from langchain_core.runnables.history import RunnableWithMessageHistory
  Use session_id to scope the history.
"""

def conversation_with_memory() -> list:
    """Runs a 3-turn conversation and returns the full message history."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    store = {}
        
    def get_session_history(session_id: str):
      if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
      return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "alex_session"}}

    turn1 = "My name is Alex. What is machine learning?"
    turn2 = "Can you give me a real-world example?"
    turn3 = "What is my name?"

    chain_with_history.invoke({"input": turn1}, config=config)
    chain_with_history.invoke({"input": turn2}, config=config)
    chain_with_history.invoke({"input": turn3}, config=config)

    history_messages = get_session_history("alex_session").messages
        
    formatted_history = []
    for msg in history_messages:
      role = "human" if msg.type == "human" else "ai"
      formatted_history.append((role, msg.content))

    return formatted_history

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 4 — Agent with Custom Tools
# ─────────────────────────────────────────────────────────────
"""
TASK 4: Agent with Custom Tools
---------------------------------
Create a LangChain agent that uses two custom tools:
  Tool 1 — word_count(text: str) → int
            Returns the number of words in a text.
  Tool 2 — reverse_text(text: str) → str
            Returns the text reversed word-by-word.

Build the agent using the @tool decorator and
create_react_agent, then run it with AgentExecutor.

Test query:
  "How many words are in 'The quick brown fox'?
   Also reverse it."

HINT:
  from langchain.agents import create_react_agent, AgentExecutor
  from langchain.tools import tool
  from langchain import hub
  prompt = hub.pull("hwchase17/react")
"""
@tool
def word_count(text: str) -> int:
  """Returns the number of words in a text."""
  return len(text.split())

@tool
def reverse_text(text: str) -> str:
  """Returns the text reversed word-by-word."""
  return " ".join(text.split()[::-1])

def agent_with_tools(query: str) -> str:
    """Runs a ReAct agent with custom tools and returns the final answer."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [word_count, reverse_text]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
        
    agent_executor = AgentExecutor(
      agent=agent, 
      tools=tools, 
      verbose=False, 
      handle_parsing_errors=True
    )
        
    result = agent_executor.invoke({"input": query})
    return result
  
    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION B — Embeddings  (Tasks 5 – 8)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 5 — Generate and Inspect Embeddings
# ─────────────────────────────────────────────────────────────
"""
TASK 5: Generate and Inspect Embeddings
-----------------------------------------
Use OpenAIEmbeddings (text-embedding-3-small) to embed a list
of sentences. Return a dict with:
  {
    "num_sentences" : int,
    "embedding_dim" : int,
    "first_5_values": list[float],   # first 5 values of sentence[0]
    "vectors"       : list[list[float]]
  }

sentences = [
  "LangChain simplifies LLM application development.",
  "pgvector adds vector search to PostgreSQL.",
  "RAG grounds language models with external knowledge.",
]

HINT:
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vectors = embeddings.embed_documents(sentences)
  A single vector is a plain Python list of floats.
"""

def generate_embeddings(sentences: list) -> dict:
    """Embeds a list of sentences and returns metadata + vectors."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    if not sentences:
      return {
        "num_sentences": 0,
        "embedding_dim": 0,
        "first_5_values": [],
        "vectors": []
      }
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    vectors = embeddings.embed_documents(sentences)
    
    num_sentences = len(vectors)
    embedding_dim = len(vectors[0])
    first_5_values = vectors[0][:5]

    return {
        "num_sentences": num_sentences,
        "embedding_dim": embedding_dim,
        "first_5_values": first_5_values,
        "vectors": vectors
    }


    # ── END OF YOUR CODE ─────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SECTION C — pgvector  (Tasks 9 – 13)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 9 — Create pgvector Table via psycopg2
# ─────────────────────────────────────────────────────────────
"""
TASK 9: Create pgvector Table via psycopg2
-------------------------------------------
Connect directly to PostgreSQL using psycopg2 and:
  1. Enable the pgvector extension.
  2. Drop then recreate a table called "documents" with:
       id       SERIAL PRIMARY KEY
       content  TEXT
       metadata JSONB
       embedding vector(1536)
  3. Return True on success, raise on error.

Prereq — PostgreSQL must be running with pgvector installed:
  CREATE EXTENSION IF NOT EXISTS vector;

HINT:
  import psycopg2, json
  conn = psycopg2.connect(os.environ["PG_CONNECTION_STRING_RAW"])
  # PG_CONNECTION_STRING_RAW = "host=... dbname=... user=... password=..."
  cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          content TEXT,
          metadata JSONB,
          embedding vector(1536)
      )
  ''')
"""

def setup_pgvector_table() -> bool:
    """Creates the pgvector extension and documents table. Returns True on success."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    conn = None
    cursor = None

    conn = psycopg2.connect(os.environ.get("PG_CONNECTION_STRING"))
    cursor = conn.cursor()
    
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("DROP TABLE IF EXISTS documents;")
    
    cursor.execute("""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector(1536)
        ); 
    """)
        
    conn.commit()
    return True

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 10 — Insert Document Embeddings into pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 10: Insert Document Embeddings
--------------------------------------
Given a list of (content, metadata) tuples, embed each document
using OpenAIEmbeddings and insert them into the "documents"
table created in Task 9.  Return the count of inserted rows.

documents = [
  ("LangChain enables LLM pipelines.", {"source": "docs", "page": 1}),
  ("pgvector stores vector embeddings.", {"source": "docs", "page": 2}),
  ("RAG retrieves relevant context.",   {"source": "paper", "page": 5}),
  ("LangSmith traces LLM calls.",       {"source": "blog",  "page": 1}),
]

HINT:
  import json
  vector = embeddings.embed_query(content)
  # Convert list to string for psycopg2:  str(vector)  or  json.dumps(vector)
  cursor.execute(
      "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
      (content, json.dumps(metadata), str(vector))
  )
"""

def insert_documents(documents: list) -> int:
    """Embeds and inserts documents. Returns count of inserted rows."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    conn = None
    cursor = None
    
    if not documents:
      return 0
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    conn = psycopg2.connect(os.environ.get("PG_CONNECTION_STRING"))
    cursor = conn.cursor()
        
    inserted_count = 0
        
    for content, metadata in documents:
      vector = embeddings.embed_query(content)
      cursor.execute(
          "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
          (content, json.dumps(metadata), str(vector))
      )
      inserted_count += 1
            
    conn.commit()
    return inserted_count

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 11 — Similarity Search with pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 11: Similarity Search with pgvector
------------------------------------------
Embed a query string and find the top-k most similar documents
using cosine distance (<=>).  Return a list of dicts:
  [{"content": str, "metadata": dict, "distance": float}, ...]

HINT:
  vector_str = str(embeddings.embed_query(query))
  cursor.execute('''
      SELECT content, metadata, embedding <=> %s AS distance
      FROM documents
      ORDER BY distance ASC
      LIMIT %s
  ''', (vector_str, top_k))
  rows = cursor.fetchall()
"""

def similarity_search(query: str, top_k: int = 3) -> list:
    """Returns top-k similar documents for the query."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    conn = None
    cursor = None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    vector_str = str(embeddings.embed_query(query))
            
    conn = psycopg2.connect(os.environ.get("PG_CONNECTION_STRING"))
    cursor = conn.cursor()
        
    cursor.execute('''
        SELECT content, metadata, embedding <=> %s AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s
        ''', (vector_str, top_k)
    )
        
    rows = cursor.fetchall()
        
    results = []
    for row in rows:
      content, metadata, distance = row
      results.append({
        "content": content,
        "metadata": metadata, 
        "distance": float(distance)
      })
            
    return results

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 12 — Metadata Filtering in pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 12: Metadata Filtering
------------------------------
Extend the similarity search to filter by a metadata field.
Only return documents whose metadata->>'source' matches the
given source value.

Example:
  results = filtered_search("LLM tracing", source_filter="blog", top_k=2)

HINT:
  Add a WHERE clause using JSONB operators:
  WHERE metadata->>'source' = %s
  Parameters: (vector_str, source_filter, top_k)
"""

def filtered_search(query: str, source_filter: str, top_k: int = 3) -> list:
    """Returns top-k similar docs filtered by metadata source."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    conn = None
    cursor = None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_str = str(embeddings.embed_query(query))
        
    conn = psycopg2.connect(os.environ.get("PG_CONNECTION_STRING"))
    cursor = conn.cursor()
        
    cursor.execute('''
        SELECT content, metadata, embedding <=> %s AS distance
        FROM documents
        WHERE metadata->>'source' = %s
        ORDER BY distance ASC
        LIMIT %s
        ''', (vector_str, source_filter, top_k)
    )
        
    rows = cursor.fetchall()
    results = []
    for row in rows:
      content, metadata, distance = row
      results.append({
        "content": content,
        "metadata": metadata, 
        "distance": float(distance)
      })
            
    return results

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 13 — LangChain PGVector VectorStore Integration
# ─────────────────────────────────────────────────────────────
"""
TASK 13: LangChain PGVector VectorStore
-----------------------------------------
Use LangChain's built-in PGVector vectorstore to:
  1. Create a PGVector store from the document list in Task 10.
  2. Run a similarity_search_with_score for a query.
  3. Return a list of (Document, score) tuples.

Use collection_name="lc_documents".

HINT:
  from langchain_community.vectorstores import PGVector
  from langchain_core.documents import Document

  docs = [Document(page_content=c, metadata=m) for c, m in documents]

  store = PGVector.from_documents(
      documents=docs,
      embedding=embeddings,
      collection_name="lc_documents",
      connection_string=os.environ["PG_CONNECTION_STRING"],
  )
  results = store.similarity_search_with_score(query, k=top_k)
"""

def langchain_pgvector_search(documents: list, query: str, top_k: int = 3) -> list:
    """Creates a PGVector store and runs a scored similarity search."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    if not documents:
      return []

    docs = [Document(page_content=c, metadata=m) for c, m in documents]
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    store = PGVector.from_documents(
                  documents=docs,
                  embedding=embeddings,
                  collection_name="lc_documents",
                  connection_string=os.environ.get("PG_CONNECTION_STRING"),
            )
    results = store.similarity_search_with_score(query, k=top_k)
    return results

    # ── END OF YOUR CODE ─────────────────────────────────────
  
# ─────────────────────────────────────────────────────────────
# SECTION D — RAG Agents  (Tasks 14 – 17)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 14 — Basic RAG Pipeline
# ─────────────────────────────────────────────────────────────
"""
TASK 14: Basic RAG Pipeline
------------------------------
Build an end-to-end RAG chain that:
  1. Loads documents from a list of strings.
  2. Stores them in a PGVector vectorstore.
  3. Creates a retriever (top-3 results).
  4. Passes retrieved context + question to ChatOpenAI.
  5. Returns the final answer string.

Use the LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

HINT:
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  prompt = ChatPromptTemplate.from_template(
      "Answer using only this context:\n{context}\n\nQuestion: {question}"
  )
"""

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def basic_rag_pipeline(documents: list, question: str) -> str:
    """Indexes documents and answers the question using RAG."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector.from_texts(
        texts=documents,
        embedding=embeddings,
        connection_string=os.environ.get("PG_CONNECTION_STRING"),
        collection_name="lc_documents"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 15 — RAG with Source Attribution
# ─────────────────────────────────────────────────────────────
"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }

HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:

  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""

def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector.from_texts(
        texts=documents,
        embedding=embeddings,
        connection_string=os.environ.get("PG_CONNECTION_STRING"),
        collection_name="lc_documents"
    )

    def retrieve_with_scores(query: str):
        return vectorstore.similarity_search_with_score(query, k=3)

    def format_docs(docs_and_scores):
        return "\n\n".join(doc.page_content for doc, score in docs_and_scores)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retrieval_step = RunnableParallel(
        docs_and_scores=retrieve_with_scores,
        question=RunnablePassthrough()
    )

    generation_step = (
        {
            "context": lambda x: format_docs(x["docs_and_scores"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = retrieval_step.assign(answer=generation_step)

    response = rag_chain.invoke(question)

    sources = [
        {"content": doc.page_content, "score": float(score)} 
        for doc, score in response["docs_and_scores"]
    ]

    return {
        "answer": response["answer"],
        "sources": sources
    }

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""

def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector.from_texts(
        texts=documents,
        embedding=embeddings,
        connection_string=os.environ.get("PG_CONNECTION_STRING"),
        collection_name="lc_documents"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []
    
    query1 = "What is LangChain?"
    response1 = rag_chain.invoke({"input": query1, "chat_history": chat_history})
    answer1 = response1["answer"]
    
    chat_history.extend(
        [HumanMessage(content=query1), AIMessage(content=answer1)]
    )

    query2 = "What version introduced LCEL?"
    response2 = rag_chain.invoke({"input": query2, "chat_history": chat_history})
    answer2 = response2["answer"]

    return [answer1, answer2]

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 17 — RAG Agent (Tool-based Retrieval)
# ─────────────────────────────────────────────────────────────
"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.

Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.

HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""

def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector.from_texts(
        texts=RAG_DOCUMENTS,
        embedding=embeddings,
        connection_string=os.environ.get("PG_CONNECTION_STRING"),
        collection_name="lc_documents"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Search the knowledge base for technical info. Always use this tool when you need to answer questions about pgvector, LangChain, or LangSmith."
    )
    
    tools = [retriever_tool]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    response = agent_executor.invoke({"input": question})
    
    return response["output"]
    # ── END OF YOUR CODE ─────────────────────────────────────
