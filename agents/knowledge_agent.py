"""
OperaMind — Knowledge Agent
=============================
Framework : LangGraph 0.4+ + LlamaIndex (RAG) for document retrieval
Model     : claude-sonnet-4-6  (generation) + text-embedding-3-small (embeddings)
Purpose   : Learns from company docs/SOPs/policies, answers employee questions
            instantly, provides internal search, guides through complex workflows.

pip install langgraph langchain-anthropic langchain-openai llama-index
            llama-index-vector-stores-chroma chromadb anthropic python-dotenv

ENV: ANTHROPIC_API_KEY, OPENAI_API_KEY (for embeddings), NOTION_TOKEN
"""
from __future__ import annotations
import os, json, datetime
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4096, temperature=0.1)

class KnowledgeState(TypedDict):
    messages: Annotated[list, add_messages]
    company_name: str
    employee_role: str
    knowledge_base_loaded: bool
    sources_cited: list[str]

# ── Vector store + RAG setup ───────────────────────────────────────────────
# In production, this index is pre-built from your Notion/Google Drive/Confluence
# and persisted in ChromaDB / Pinecone / Weaviate.
# Below is the setup pattern — actual ingestion runs during onboarding.

def build_knowledge_index(documents_dir: str = "./company_docs"):
    """
    Build the vector index from company documents.
    Call this once during setup, or incrementally as docs change.

    Supports: PDF, DOCX, Notion pages, Google Drive, Confluence, SharePoint
    """
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core import Settings
        from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.llm = LlamaAnthropic(model="claude-sonnet-4-6")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        docs_path = Path(documents_dir)
        if not docs_path.exists():
            docs_path.mkdir(parents=True)
            print(f"[KNOWLEDGE] Created docs directory: {docs_path}")
            return None

        reader = SimpleDirectoryReader(str(docs_path), recursive=True)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        print(f"[KNOWLEDGE] Indexed {len(documents)} documents from {documents_dir}")
        return index
    except ImportError:
        print("[KNOWLEDGE] LlamaIndex not installed. Install with: pip install llama-index")
        return None

# ── Tools ──────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(
    query: str,
    max_results: int = 5,
    filter_category: str = "",
) -> dict:
    """
    Search the company knowledge base (SOPs, policies, handbooks, product docs).
    Returns relevant passages with source citations.
    """
    print(f"[SEARCH] Searching knowledge base: '{query}'")
    # Production: query ChromaDB / Pinecone with the vector index
    # retriever = index.as_retriever(similarity_top_k=max_results)
    # nodes = retriever.retrieve(query)

    # Simulated results for demonstration
    mock_results = [
        {
            "source": "Employee Handbook v3.2 — Section 4",
            "content": f"Relevant policy found for query '{query}': Standard procedure is to submit requests through the HR portal with 3 business days notice. Manager approval required for amounts over $500.",
            "relevance_score": 0.94,
            "last_updated": "2025-03-01",
        },
        {
            "source": "SOP-OPS-012: Expense Reimbursement Process",
            "content": "Employees must submit expense reports within 30 days of the purchase. Receipts required for all items over $25. Submit via Notion form → Finance queue.",
            "relevance_score": 0.88,
            "last_updated": "2025-01-15",
        },
    ]

    return {
        "query": query,
        "results": mock_results[:max_results],
        "total_found": len(mock_results),
        "knowledge_base_version": "2025.06",
        "searched_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def get_sop_steps(
    process_name: str,
    employee_role: str = "",
) -> dict:
    """
    Retrieve the step-by-step SOP for a specific business process.
    Returns structured steps the employee can follow.
    """
    print(f"[SOP] Fetching SOP for: {process_name}")
    # Production: query the SOP database by name/category
    sop_database = {
        "onboarding": {
            "title": "New Employee Onboarding",
            "estimated_time": "3 hours",
            "steps": [
                "Complete IT setup request (IT form in Notion)",
                "Sign NDA and employment contract (DocuSign link in welcome email)",
                "Complete benefits enrollment (HR portal — deadline: 5 business days)",
                "Schedule 1:1 with direct manager (use their Calendly link)",
                "Complete security awareness training (Courses folder in Google Drive)",
                "Get added to all relevant Slack channels (ask #it-help)",
                "Join weekly team standup (recurring invite sent by manager)",
            ],
            "owner": "HR Team",
            "last_reviewed": "2025-04-01",
        },
        "expense_reimbursement": {
            "title": "Expense Reimbursement",
            "estimated_time": "10 minutes",
            "steps": [
                "Collect all receipts (digital or photo)",
                "Open the Expense Report template in Notion",
                "Fill in merchant, amount, date, and GL category for each expense",
                "Attach receipt photos",
                "Submit to Finance queue (button at bottom of form)",
                "Your manager will approve within 2 business days",
                "Finance processes payment within 5 business days of approval",
            ],
            "owner": "Finance Team",
            "last_reviewed": "2025-05-10",
        },
    }

    process_key = process_name.lower().replace(" ", "_").replace("-", "_")
    sop = sop_database.get(process_key)

    if not sop:
        return {
            "process": process_name,
            "found": False,
            "suggestion": "Try searching the knowledge base with more specific terms.",
        }

    return {"process": process_name, "found": True, "sop": sop}

@tool
def ingest_new_document(
    document_url: str,
    document_type: Literal["notion_page", "google_doc", "pdf", "confluence", "word"],
    category: str,
    added_by: str,
) -> dict:
    """
    Add a new document to the knowledge base.
    The agent will fetch, chunk, embed, and index it.
    """
    print(f"[INGEST] Adding {document_type} from {document_url}")
    # Production: fetch doc → chunk → embed → upsert to vector store
    return {
        "ingested": True,
        "document_url": document_url,
        "document_type": document_type,
        "category": category,
        "chunks_created": 12,  # estimated
        "available_immediately": True,
        "indexed_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def identify_knowledge_gaps(
    unanswered_queries: list[str],
) -> dict:
    """
    Analyse recent questions the agent couldn't fully answer.
    Returns a report for knowledge base maintainers.
    """
    print(f"[GAPS] Analysing {len(unanswered_queries)} unanswered queries")
    return {
        "total_gaps": len(unanswered_queries),
        "top_missing_topics": unanswered_queries[:5],
        "recommended_actions": [
            f"Create SOP for: {q}" for q in unanswered_queries[:3]
        ],
        "report_sent_to": "knowledge-managers@operamind.ai",
        "analysed_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def get_company_policy(
    policy_name: str,
) -> dict:
    """
    Retrieve a specific company policy by name.
    Examples: 'remote work', 'PTO', 'travel', 'security', 'code of conduct'
    """
    print(f"[POLICY] Fetching policy: {policy_name}")
    policies = {
        "pto": {
            "name": "PTO & Leave Policy",
            "summary": "Full-time employees accrue 20 days PTO per year. Unused PTO up to 10 days rolls over. Submit via HR portal with 2 weeks notice for 3+ consecutive days.",
            "full_doc_url": "https://notion.so/policies/pto",
            "effective_date": "2025-01-01",
        },
        "remote work": {
            "name": "Remote Work Policy",
            "summary": "Employees may work remotely up to 3 days/week. Core hours 10am-3pm local time. In-office required for team meetings and quarterly planning.",
            "full_doc_url": "https://notion.so/policies/remote",
            "effective_date": "2024-09-01",
        },
    }
    key = policy_name.lower().strip()
    return policies.get(key, {
        "policy": policy_name,
        "found": False,
        "suggestion": "Search the knowledge base or contact HR for this policy.",
    })

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are OperaMind's AI Knowledge Employee for {company_name}.

You are the company's institutional memory — you know every policy, SOP, and process.
Your job is to help employees get instant, accurate answers without hunting through docs.

How to respond:
1. Always search the knowledge base FIRST before answering
2. For process questions: provide step-by-step SOP guidance
3. For policy questions: cite the exact policy and link to the full document
4. If you don't find a confident answer: say so clearly and suggest who to contact
5. Always cite your source (document name + last updated date)
6. Flag if a document seems outdated (> 6 months since last review)

Tone: Helpful, clear, concise. Like a knowledgeable senior colleague, not a legal document.
Never guess. Always cite sources. Always be accurate.

Company: {company_name}
Employee role context: {employee_role}
"""

# ── Graph ──────────────────────────────────────────────────────────────────

tools = [search_knowledge_base, get_sop_steps, ingest_new_document,
         identify_knowledge_gaps, get_company_policy]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def knowledge_agent(state: KnowledgeState) -> KnowledgeState:
    system = SYSTEM_PROMPT.format(
        company_name=state.get("company_name", "the Company"),
        employee_role=state.get("employee_role", "team member"),
    )
    response = llm_with_tools.invoke([SystemMessage(content=system)] + state["messages"])
    return {"messages": [response]}

def router(state: KnowledgeState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

checkpointer = MemorySaver()
builder = StateGraph(KnowledgeState)
builder.add_node("agent", knowledge_agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)

# ── Public API ─────────────────────────────────────────────────────────────

def ask(
    question: str,
    company_name: str,
    employee_role: str = "employee",
    session_id: str | None = None,
) -> dict:
    """
    Ask the knowledge agent a question.

    Returns answer with source citations.
    """
    sid = session_id or f"kb_{datetime.datetime.utcnow().timestamp():.0f}"
    config = {"configurable": {"thread_id": sid}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)],
         "company_name": company_name, "employee_role": employee_role,
         "knowledge_base_loaded": True, "sources_cited": []},
        config=config,
    )
    final = next((m for m in reversed(result["messages"])
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {
        "question": question,
        "answer": final.content if final else "I couldn't find a confident answer. Please contact HR.",
        "session_id": sid,
    }

def add_document(
    document_url: str,
    document_type: str,
    category: str,
    added_by: str,
    company_name: str,
) -> dict:
    """Add a new document to the knowledge base."""
    config = {"configurable": {"thread_id": f"ingest_{datetime.datetime.utcnow().timestamp():.0f}"}}
    prompt = f"Please ingest this document into the knowledge base: {document_url} (type: {document_type}, category: {category}, added by: {added_by})"
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)],
         "company_name": company_name, "employee_role": "admin",
         "knowledge_base_loaded": True, "sources_cited": []},
        config=config,
    )
    return {"ingested": True, "document": document_url}


if __name__ == "__main__":
    # Example: employee asks about expense reimbursement
    result = ask(
        question="How do I submit an expense reimbursement? I have $340 in receipts from a client dinner.",
        company_name="Acme Corp",
        employee_role="Sales Manager",
    )
    print("\nKNOWLEDGE AGENT ANSWER:\n", result["answer"])

    # Example: HR policy question
    result2 = ask(
        question="How many PTO days do I get and can I roll them over?",
        company_name="Acme Corp",
        employee_role="Software Engineer",
        session_id=result["session_id"],  # same session
    )
    print("\nFOLLOW-UP ANSWER:\n", result2["answer"])
