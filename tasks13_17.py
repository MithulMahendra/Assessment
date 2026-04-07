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
