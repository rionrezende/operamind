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

@tool
def check_document_freshness(
    document_name: str,
    last_updated: str,
) -> dict:
    """
    Check whether a document is still fresh or needs review.
    Takes the document name and its last_updated date (ISO format, e.g. '2025-01-15').
    Returns freshness status and recommendation.
    """
    print(f"[FRESHNESS] Checking freshness for: {document_name}")
    today = datetime.date.today()
    updated_date = datetime.date.fromisoformat(last_updated)
    age_days = (today - updated_date).days

    if age_days > 365:
        freshness_status = "expired"
        recommendation = f"'{document_name}' is over a year old. Immediate review and update required."
    elif age_days > 180:
        freshness_status = "stale"
        recommendation = f"'{document_name}' is over 6 months old. Schedule a review with the document owner."
    elif age_days > 90:
        freshness_status = "aging"
        recommendation = f"'{document_name}' is aging. Consider reviewing within the next month."
    else:
        freshness_status = "current"
        recommendation = f"'{document_name}' is up to date. No action needed."

    return {
        "document_name": document_name,
        "is_stale": age_days > 180,
        "age_days": age_days,
        "last_updated": last_updated,
        "freshness_status": freshness_status,
        "recommendation": recommendation,
    }

@tool
def score_answer_confidence(
    answer_text: str,
    sources_found: int,
    source_relevance_avg: float,
) -> dict:
    """
    Score the confidence of an answer based on available sources.
    source_relevance_avg should be a float between 0 and 1.
    Returns confidence level, score, and whether the answer should be verified.
    """
    print(f"[CONFIDENCE] Scoring answer confidence (sources={sources_found}, relevance={source_relevance_avg:.2f})")

    # Base score from number of sources
    if sources_found == 0:
        base_score = 10
    elif sources_found <= 2:
        base_score = 50
    else:
        base_score = 80

    # Weight with relevance (0-1 scale)
    relevance_clamped = max(0.0, min(1.0, source_relevance_avg))
    confidence_score = int(base_score * (0.4 + 0.6 * relevance_clamped))
    confidence_score = max(0, min(100, confidence_score))

    if confidence_score >= 75:
        confidence_level = "high"
    elif confidence_score >= 40:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    should_verify = confidence_score < 60

    result = {
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "should_verify": should_verify,
        "sources_found": sources_found,
        "source_relevance_avg": source_relevance_avg,
    }

    if confidence_level == "low":
        result["disclaimer"] = (
            "This answer has low confidence. The information may be incomplete or outdated. "
            "Please verify with the relevant department or document owner before acting on it."
        )

    return result

@tool
def generate_faq(
    topic: str,
    recent_questions: list[str],
) -> dict:
    """
    Generate a FAQ from recent employee questions on a topic.
    Groups similar questions, picks the top 5, and provides answers.
    """
    print(f"[FAQ] Generating FAQ for topic: {topic} ({len(recent_questions)} questions)")

    # Group similar questions by simple keyword overlap
    seen = []
    grouped: list[dict] = []
    for q in recent_questions:
        q_lower = q.lower()
        matched = False
        for g in grouped:
            # Simple similarity: check if most words overlap
            g_words = set(g["representative"].lower().split())
            q_words = set(q_lower.split())
            overlap = len(g_words & q_words) / max(len(g_words | q_words), 1)
            if overlap > 0.4:
                g["times_asked"] += 1
                matched = True
                break
        if not matched:
            grouped.append({"representative": q, "times_asked": 1})

    # Sort by frequency, take top 5
    grouped.sort(key=lambda g: g["times_asked"], reverse=True)
    top_questions = grouped[:5]

    # Generate mock answers
    faq_items = []
    categories = ["General", "Process", "Policy", "Benefits", "Technical"]
    for i, item in enumerate(top_questions):
        faq_items.append({
            "question": item["representative"],
            "answer": f"Regarding '{item['representative']}': Please refer to the company knowledge base for the most current information on {topic}. If you need further assistance, contact the relevant department.",
            "category": categories[i % len(categories)],
            "times_asked": item["times_asked"],
        })

    return {
        "topic": topic,
        "total_questions_analysed": len(recent_questions),
        "faq": faq_items,
        "generated_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def create_onboarding_path(
    new_hire_name: str,
    role: str,
    department: str,
    start_date: str,
) -> dict:
    """
    Create a personalized 30-day onboarding checklist for a new hire.
    start_date should be in ISO format (e.g. '2025-07-01').
    Returns a week-by-week onboarding plan with tasks, responsible persons, and completion criteria.
    """
    print(f"[ONBOARDING] Creating onboarding path for {new_hire_name} ({role}, {department})")

    onboarding_plan = {
        "new_hire": new_hire_name,
        "role": role,
        "department": department,
        "start_date": start_date,
        "weeks": {
            "week_1": {
                "theme": "Setup & Orientation",
                "tasks": [
                    "Complete IT setup (laptop, accounts, access permissions)",
                    "Sign all employment documents (NDA, contract, benefits)",
                    "Attend company orientation session",
                    f"Meet with {department} team lead for role overview",
                    "Set up development environment and tools" if "engineer" in role.lower() or "developer" in role.lower() else f"Set up {department}-specific tools and software",
                ],
                "responsible_person": "HR Coordinator + IT Support",
                "completion_criteria": "All accounts active, documents signed, orientation attended",
            },
            "week_2": {
                "theme": "Learning & Integration",
                "tasks": [
                    f"Shadow a senior {role} for daily workflow",
                    "Complete mandatory compliance training modules",
                    "Review team documentation and SOPs",
                    "Attend first team standup and weekly meeting",
                    "Have 1:1 with direct manager to set 30-day goals",
                ],
                "responsible_person": "Direct Manager + Assigned Buddy",
                "completion_criteria": "Training modules completed, 30-day goals documented",
            },
            "week_3": {
                "theme": "Hands-On Contribution",
                "tasks": [
                    "Take on first independent task or project",
                    "Present learnings to the team (informal)",
                    "Review and provide feedback on onboarding experience so far",
                    f"Connect with cross-functional partners in related departments",
                    "Complete role-specific certification if applicable",
                ],
                "responsible_person": "Direct Manager",
                "completion_criteria": "First task completed, cross-functional introductions made",
            },
            "week_4": {
                "theme": "Autonomy & Review",
                "tasks": [
                    "Complete 30-day self-assessment",
                    "Have formal 30-day review with manager",
                    "Identify areas for continued learning",
                    "Set 60-day and 90-day goals with manager",
                    "Provide written feedback on onboarding process to HR",
                ],
                "responsible_person": "Direct Manager + HR",
                "completion_criteria": "30-day review completed, 90-day goals set, feedback submitted",
            },
        },
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    return {"onboarding_plan": onboarding_plan}

@tool
def notify_policy_change(
    document_name: str,
    change_summary: str,
    changed_by: str,
    affected_departments: list[str],
) -> dict:
    """
    Notify relevant teams about a policy or document change.
    Identifies who needs to know based on affected departments and sends notifications.
    """
    print(f"[NOTIFY] Policy change notification for: {document_name}")

    notification_list = []
    for dept in affected_departments:
        # Determine notification method and urgency based on department
        urgency = "high" if dept.lower() in ("legal", "compliance", "security", "finance") else "normal"
        method = "slack" if urgency == "normal" else "email"

        notification_list.append({
            "department": dept,
            "notification_method": method,
            "urgency": urgency,
            "sent": True,
            "sent_at": datetime.datetime.utcnow().isoformat(),
        })

    return {
        "document_name": document_name,
        "change_summary": change_summary,
        "changed_by": changed_by,
        "total_departments_notified": len(notification_list),
        "notification_list": notification_list,
        "notified_at": datetime.datetime.utcnow().isoformat(),
    }

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
6. Flag if a document seems outdated — use check_document_freshness to verify
7. After answering, use score_answer_confidence to assess answer quality; include a disclaimer if confidence is low
8. For recurring questions on a topic, use generate_faq to create a FAQ summary
9. For new hire questions, use create_onboarding_path to generate a personalized 30-day plan
10. When a policy changes, use notify_policy_change to alert affected departments

Tone: Helpful, clear, concise. Like a knowledgeable senior colleague, not a legal document.
Never guess. Always cite sources. Always be accurate.

Company: {company_name}
Employee role context: {employee_role}
"""

# ── Graph ──────────────────────────────────────────────────────────────────

tools = [search_knowledge_base, get_sop_steps, ingest_new_document,
         identify_knowledge_gaps, get_company_policy, check_document_freshness,
         score_answer_confidence, generate_faq, create_onboarding_path,
         notify_policy_change]
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
