"""
OperaMind — Customer Success Agent
====================================
Framework : LangGraph 0.4+ (stateful, HITL escalation, durable)
Model     : claude-sonnet-4-6
Purpose   : Resolves 70-80% of support tickets autonomously.
            Processes refunds, creates tickets, escalates complex cases
            with full context, sends follow-ups.

pip install langgraph langchain-anthropic anthropic python-dotenv pydantic

ENV: ANTHROPIC_API_KEY, ZENDESK_API_KEY, ZENDESK_SUBDOMAIN, SHOPIFY_ACCESS_TOKEN
"""
from __future__ import annotations
import os, datetime
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

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048, temperature=0.1)

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    ticket: dict
    customer: dict
    resolution: str | None        # 'resolved' | 'escalated' | 'pending'
    confidence: float             # agent's confidence 0-1
    actions_taken: list[str]

# ── Tools ─────────────────────────────────────────────────────────────────

@tool
def lookup_order(order_id: str, customer_email: str) -> dict:
    """
    Look up an order from Shopify / WooCommerce by order ID or customer email.
    Returns order status, items, total, and fulfilment details.
    """
    # Production: GET https://{shop}.myshopify.com/admin/api/2024-01/orders/{id}.json
    print(f"[SHOPIFY] Looking up order {order_id} for {customer_email}")
    return {
        "order_id": order_id,
        "customer_email": customer_email,
        "status": "fulfilled",
        "total": 149.00,
        "currency": "USD",
        "items": [{"name": "OperaMind Starter Plan", "qty": 1, "price": 149.00}],
        "created_at": "2025-06-01T10:00:00Z",
        "fulfillment_status": "delivered",
        "days_since_order": 5,
        "within_refund_policy": True,   # < 30 days
    }

@tool
def lookup_customer(email: str) -> dict:
    """Fetch customer profile including tier, lifetime value, and history."""
    print(f"[CRM] Looking up customer {email}")
    return {
        "email": email,
        "name": "Sarah Mitchell",
        "tier": "Premium",
        "lifetime_value": 1840.00,
        "total_orders": 12,
        "open_tickets": 0,
        "csat_average": 4.8,
        "member_since": "2023-11-01",
        "risk_of_churn": "low",
    }

@tool
def process_refund(
    order_id: str,
    amount: float,
    reason: str,
    send_confirmation: bool = True,
) -> dict:
    """
    Process a full or partial refund via Stripe / payment processor.
    Only call after confirming the order is within refund policy.
    """
    print(f"[PAYMENTS] Processing refund of ${amount} for order {order_id}. Reason: {reason}")
    return {
        "success": True,
        "refund_id": f"ref_{order_id}_{datetime.datetime.utcnow().timestamp():.0f}",
        "amount_refunded": amount,
        "estimated_arrival": "3–5 business days",
        "confirmation_sent": send_confirmation,
    }

@tool
def create_ticket(
    customer_email: str,
    subject: str,
    description: str,
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
    tags: list[str] | None = None,
) -> dict:
    """Create a support ticket in Zendesk / Freshdesk."""
    print(f"[ZENDESK] Creating {priority} ticket for {customer_email}: {subject}")
    return {
        "ticket_id": f"TKT-{datetime.datetime.utcnow().microsecond}",
        "status": "open",
        "priority": priority,
        "assignee": "AI Support Agent",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def escalate_to_human(
    ticket_id: str,
    reason: str,
    context_summary: str,
    recommended_action: str,
    urgency: Literal["low", "normal", "high"] = "normal",
) -> dict:
    """
    Escalate a ticket to a human agent with full context.
    Use when: complex disputes, legal threats, technical issues beyond scope,
    high-value customer at churn risk, or confidence < 0.6.
    """
    print(f"[ESCALATION] Ticket {ticket_id} → human agent. Reason: {reason}")
    print(f"  Context: {context_summary}")
    print(f"  Recommended action: {recommended_action}")
    return {
        "escalated": True,
        "assigned_to": "Senior Support Team",
        "estimated_response": "2 hours",
        "context_preserved": True,
    }

@tool
def send_support_email(
    to_email: str,
    customer_name: str,
    email_type: Literal["resolution", "refund_confirmed", "escalation_notice",
                        "followup", "loyalty_offer", "apology"],
    ticket_id: str,
    custom_message: str = "",
    voucher_code: str | None = None,
) -> dict:
    """Send a templated support email to the customer."""
    templates = {
        "resolution": f"Hi {customer_name}, your case has been resolved.",
        "refund_confirmed": f"Hi {customer_name}, your refund has been processed.",
        "escalation_notice": f"Hi {customer_name}, a senior agent is reviewing your case.",
        "apology": f"Hi {customer_name}, we sincerely apologize for the inconvenience.",
    }
    body = templates.get(email_type, custom_message)
    if voucher_code:
        body += f"\n\nAs a token of appreciation, here's a 15% discount code: {voucher_code}"
    print(f"[EMAIL] Sending {email_type} email to {customer_name} <{to_email}>")
    return {
        "sent": True,
        "email_type": email_type,
        "voucher_included": bool(voucher_code),
        "sent_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def update_ticket_status(
    ticket_id: str,
    status: Literal["open", "pending", "solved", "closed"],
    resolution_notes: str,
    csat_survey: bool = True,
) -> dict:
    """Update ticket status and optionally trigger a CSAT survey."""
    print(f"[ZENDESK] Ticket {ticket_id} → {status}. Notes: {resolution_notes}")
    return {
        "ticket_id": ticket_id,
        "new_status": status,
        "csat_scheduled": csat_survey,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }

# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are OperaMind's AI Customer Success Employee.

Your mission: Resolve customer issues quickly, accurately, and empathetically.
Resolve 70-80% of cases autonomously. Escalate the rest with full context.

Resolution framework:
1. Always look up the order AND the customer profile first
2. Assess whether the request is within policy (refund window, warranty, etc.)
3. If within policy and clear-cut: resolve autonomously (process refund, create ticket, etc.)
4. If complex, legal, or confidence < 0.6: escalate with full context + recommendation
5. Always send a confirmation email and update ticket status
6. For Premium/high-LTV customers: include a loyalty gesture (voucher, priority, apology)

NEVER:
- Process a refund outside policy without escalating first
- Make promises you can't keep
- Leave a ticket without a status update

Tone: Warm, empathetic, professional. Acknowledge frustration. Focus on resolution speed.
Current date: {date}
"""

# ── Graph ──────────────────────────────────────────────────────────────────

tools = [lookup_order, lookup_customer, process_refund, create_ticket,
         escalate_to_human, send_support_email, update_ticket_status]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: SupportState) -> SupportState:
    system = SYSTEM_PROMPT.format(date=datetime.date.today())
    response = llm_with_tools.invoke([SystemMessage(content=system)] + state["messages"])
    return {"messages": [response]}

def router(state: SupportState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

checkpointer = MemorySaver()
builder = StateGraph(SupportState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)

# ── Public API ─────────────────────────────────────────────────────────────

def handle_ticket(ticket: dict) -> dict:
    """
    ticket = {
        "id": "TKT-4821",
        "customer_email": "sarah@brightline.com",
        "subject": "Refund request for order ORD-9934",
        "message": "I ordered 5 days ago and haven't received what I paid for. I want a refund.",
        "channel": "email",     # email | chat | whatsapp
        "language": "en",
    }
    """
    tid = ticket.get("id", f"tkt_{datetime.datetime.utcnow().timestamp():.0f}")
    config = {"configurable": {"thread_id": tid}}
    prompt = (
        f"Support ticket received:\n"
        f"Ticket ID: {ticket.get('id')}\n"
        f"Customer: {ticket.get('customer_email')}\n"
        f"Subject: {ticket.get('subject')}\n"
        f"Message: {ticket.get('message')}\n"
        f"Channel: {ticket.get('channel', 'email')}\n\n"
        f"Please resolve this ticket following the resolution framework."
    )
    initial: SupportState = {
        "messages": [HumanMessage(content=prompt)],
        "ticket": ticket,
        "customer": {},
        "resolution": None,
        "confidence": 0.0,
        "actions_taken": [],
    }
    result = graph.invoke(initial, config=config)
    final = next((m for m in reversed(result["messages"])
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {"ticket_id": tid, "agent_response": final.content if final else "Resolved."}


if __name__ == "__main__":
    ticket = {
        "id": "TKT-4821",
        "customer_email": "sarah@brightline.com",
        "subject": "Refund for order ORD-9934",
        "message": "I've been waiting 5 days and still haven't received my order. I want a full refund.",
        "channel": "email",
    }
    r = handle_ticket(ticket)
    print("\nAGENT RESPONSE:\n", r["agent_response"])
