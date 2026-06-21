# OperaMind AI Agents

7 production-ready AI employees built on state-of-the-art frameworks.

## Quick Start

```bash
cd agents/
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python sales_agent.py  # test any agent
```

## Agent Directory

| File | Agent | Framework | Primary Tools |
|------|-------|-----------|---------------|
| `sales_agent.py` | Sales Agent | LangGraph | HubSpot, Calendly, Gmail |
| `customer_success_agent.py` | Customer Success | LangGraph | Zendesk, Shopify, Stripe |
| `recruiting_agent.py` | Recruiting Agent | CrewAI | Greenhouse, LinkedIn, Calendly |
| `finance_agent.py` | Finance Agent | CrewAI | QuickBooks, Xero, Bill.com |
| `operations_agent.py` | Operations Agent | LangGraph | Slack, Jira, Notion |
| `executive_assistant_agent.py` | Executive Assistant | LangGraph | Gmail, Google Calendar |
| `knowledge_agent.py` | Knowledge Agent | LangGraph + RAG | Notion, Confluence, Drive |

## Architecture

### Why LangGraph for most agents?
- **Stateful** — conversations persist across sessions (durable memory)
- **HITL** — interrupt at any node for human approval before taking action
- **Production-grade** — checkpointing, retries, fault tolerance built in
- **Auditable** — full execution trace in LangSmith

### Why CrewAI for Recruiting and Finance?
- Both decompose naturally into **parallel specialist roles**
- Recruiting: Screener + Interviewer + Evaluator work independently
- Finance: Extractor + Auditor + Reconciler + Reporter are separate concerns
- CrewAI's role-based model maps perfectly to these workflows

## Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...

# CRM
HUBSPOT_API_KEY=...
# or
SALESFORCE_CLIENT_ID=...
SALESFORCE_CLIENT_SECRET=...

# Support
ZENDESK_API_KEY=...
ZENDESK_SUBDOMAIN=...

# Payments
SHOPIFY_ACCESS_TOKEN=...
STRIPE_SECRET_KEY=...

# Communication
SLACK_BOT_TOKEN=xoxb-...
SENDGRID_API_KEY=...

# Productivity
GOOGLE_CREDENTIALS_JSON=...   # path to credentials file
NOTION_TOKEN=...

# Accounting
QUICKBOOKS_CLIENT_ID=...
QUICKBOOKS_CLIENT_SECRET=...

# Embeddings (Knowledge Agent)
OPENAI_API_KEY=...            # for text-embedding-3-small
```

## Deploying to Production

### Option 1: FastAPI wrapper (recommended)
```python
from fastapi import FastAPI
from sales_agent import process_lead

app = FastAPI()

@app.post("/agents/sales/lead")
async def new_lead(lead: dict):
    return process_lead(lead)
```

### Option 2: LangGraph Platform
```bash
pip install langgraph-cli
langgraph up  # deploys all graphs with persistence, streaming, and API
```

### Option 3: AWS Lambda / Cloud Functions
Each agent's public function (process_lead, handle_ticket, etc.)
is a Lambda-compatible handler.

## Persistence

All agents use `MemorySaver` by default (in-memory, resets on restart).

For production, swap with persistent checkpointers:

```python
# SQLite (single server)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./operamind.db")

# PostgreSQL (multi-server, recommended for production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
```

## Observability

All LangGraph agents are compatible with LangSmith for tracing:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "operamind-agents"
```
