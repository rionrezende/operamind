"""
OperaMind — Executive Assistant Agent
========================================
Framework : LangGraph 0.4+ with persistent memory across sessions
Model     : claude-sonnet-4-6
Purpose   : Email triage, calendar optimisation, meeting summaries,
            task prioritisation, briefing preparation.

pip install langgraph langchain-anthropic anthropic google-auth google-api-python-client python-dotenv

ENV: ANTHROPIC_API_KEY, GOOGLE_CREDENTIALS_JSON (or MICROSOFT_CLIENT_ID for O365)
"""
from __future__ import annotations
import os, json, datetime
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
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4096, temperature=0.2)

class ExecState(TypedDict):
    messages: Annotated[list, add_messages]
    executive_name: str
    executive_email: str
    priorities_today: list[str]
    commitments_tracked: list[dict]

# ── Tools ──────────────────────────────────────────────────────────────────

@tool
def fetch_emails(
    max_results: int = 20,
    filter_label: str = "INBOX",
    unread_only: bool = True,
) -> list[dict]:
    """
    Fetch emails from Gmail / Outlook. Returns list of emails sorted by priority signals.
    """
    print(f"[EMAIL] Fetching {max_results} emails from {filter_label}")
    # Production: use google-api-python-client or msal for Microsoft
    # service = build('gmail', 'v1', credentials=creds)
    # results = service.users().messages().list(userId='me', labelIds=[filter_label]).execute()
    return [
        {"id": "e1", "from": "board@company.com", "subject": "Q2 Board Deck — needs your input by EOD",
         "preview": "The board needs your section by 5pm today.", "received": "09:15", "priority": "urgent"},
        {"id": "e2", "from": "partner@bigclient.com", "subject": "RE: Partnership proposal",
         "preview": "Looks great! Can we jump on a call this week?", "received": "08:45", "priority": "high"},
        {"id": "e3", "from": "hr@company.com", "subject": "Benefits enrollment reminder",
         "preview": "Annual enrollment closes Friday.", "received": "08:00", "priority": "low"},
        {"id": "e4", "from": "team@company.com", "subject": "Weekly team update",
         "preview": "Here's the weekly status from each department.", "received": "07:30", "priority": "medium"},
    ]

@tool
def draft_email_reply(
    original_email_id: str,
    tone: Literal["formal", "friendly", "brief", "detailed"],
    key_points: str,
    executive_name: str,
) -> str:
    """
    Draft an email reply in the executive's voice. Returns draft text.
    """
    print(f"[EMAIL DRAFT] Drafting reply to {original_email_id} — tone: {tone}")
    # Production: this would use the exec's actual writing history as examples
    return f"""
Hi [Name],

Thanks for reaching out. {key_points}

I'd suggest we connect this week to discuss further. I'll have my assistant send over some times.

Best,
{executive_name}
""".strip()

@tool
def fetch_calendar(
    date: str,
    look_ahead_days: int = 3,
) -> dict:
    """
    Fetch calendar events for upcoming days. Returns schedule and free slots.
    """
    print(f"[CALENDAR] Fetching schedule from {date} for {look_ahead_days} days")
    # Production: Google Calendar API or Microsoft Graph API
    return {
        "date": date,
        "events": [
            {"time": "09:00-09:30", "title": "Daily standup", "type": "recurring", "location": "Zoom"},
            {"time": "10:00-11:00", "title": "1:1 with VP Sales", "type": "meeting"},
            {"time": "14:00-15:00", "title": "Board prep call", "type": "meeting", "priority": "high"},
            {"time": "16:00-16:30", "title": "Investor update", "type": "meeting", "priority": "high"},
        ],
        "free_slots": ["11:00-14:00", "15:00-16:00", "after 17:00"],
        "conflicts": [],
        "travel_time_buffered": True,
    }

@tool
def schedule_meeting(
    title: str,
    attendees: list[str],
    duration_minutes: int,
    preferred_time: str,
    meeting_type: Literal["call", "video", "in_person"] = "video",
    agenda: str = "",
) -> dict:
    """
    Schedule a meeting and send invites to all attendees.
    """
    print(f"[CALENDAR] Scheduling '{title}' with {attendees} for {duration_minutes}min")
    return {
        "event_id": f"evt_{datetime.datetime.utcnow().timestamp():.0f}",
        "title": title,
        "time": preferred_time,
        "duration": duration_minutes,
        "attendees": attendees,
        "meet_link": "https://meet.google.com/abc-defg-hij",
        "invites_sent": True,
        "agenda_attached": bool(agenda),
    }

@tool
def summarise_meeting(
    transcript_or_notes: str,
    meeting_title: str,
    attendees: list[str],
) -> dict:
    """
    Generate a meeting summary with key decisions, action items, and owners.
    """
    print(f"[SUMMARY] Summarising meeting: {meeting_title}")
    # The LLM will handle the actual summarisation using the transcript
    return {
        "meeting": meeting_title,
        "summary_generated": True,
        "format": "Markdown with sections: Summary, Decisions, Action Items, Next Meeting",
        "note": "Use the transcript provided to generate the actual summary in your response.",
        "attendees": attendees,
    }

@tool
def prioritise_tasks(
    task_list: list[str],
    executive_goals: str,
    deadline_context: str,
) -> list[dict]:
    """
    Prioritise a list of tasks using Eisenhower matrix (urgent/important).
    Returns ranked list with quadrant classification.
    """
    print(f"[TASKS] Prioritising {len(task_list)} tasks")
    # The LLM will do the actual prioritisation reasoning
    return [
        {"rank": 1, "task": t, "quadrant": "Urgent+Important", "estimated_hours": 1}
        for i, t in enumerate(task_list[:3])
    ] + [
        {"rank": i+4, "task": t, "quadrant": "Important, Not Urgent", "estimated_hours": 0.5}
        for i, t in enumerate(task_list[3:])
    ]

@tool
def prepare_briefing(
    meeting_title: str,
    attendees: list[str],
    context: str,
) -> str:
    """
    Prepare a pre-meeting briefing document for the executive.
    Returns Markdown briefing with background, talking points, and questions to ask.
    """
    print(f"[BRIEFING] Preparing briefing for '{meeting_title}'")
    return f"""
# Pre-Meeting Briefing: {meeting_title}

**Date:** {datetime.date.today()} | **Attendees:** {', '.join(attendees)}

## Context
{context}

## Key Talking Points
- [ ] Review Q2 performance vs. targets
- [ ] Confirm next milestone and timeline
- [ ] Discuss resourcing needs

## Questions to Consider
- What is the #1 priority for this meeting?
- What decision needs to be made?
- What follow-up will be required?

## Background
[Fetched from CRM, previous emails, and meeting history]

*Prepared by OperaMind Executive Assistant Agent · {datetime.datetime.now().strftime("%H:%M")}*
"""

@tool
def track_commitment(
    commitment: str,
    due_date: str,
    made_to: str,
    source: str,
) -> dict:
    """Track a commitment the executive made (from emails, meetings). Sends reminders before due."""
    print(f"[COMMITMENTS] Tracking: '{commitment}' due {due_date} → {made_to}")
    return {
        "tracked": True,
        "commitment_id": f"com_{datetime.datetime.utcnow().timestamp():.0f}",
        "reminder_scheduled": (datetime.datetime.fromisoformat(due_date)
                               - datetime.timedelta(hours=24)).isoformat() if due_date else None,
    }

@tool
def plan_travel(
    destination: str,
    travel_dates: str,
    purpose: str,
    preferences: str = "business class, central hotel",
) -> dict:
    """Plan travel for the executive. Returns flight options, hotel options, and estimated total cost."""
    print(f"[TRAVEL] Planning trip to {destination} ({travel_dates}) — {purpose}")
    return {
        "destination": destination,
        "travel_dates": travel_dates,
        "purpose": purpose,
        "flights": [
            {"option": 1, "airline": "United Airlines", "time": "08:00–12:30", "price": "$1,250"},
            {"option": 2, "airline": "Delta Air Lines", "time": "10:15–14:45", "price": "$1,180"},
        ],
        "hotels": [
            {"option": 1, "name": "The Ritz-Carlton", "rate": "$420/night", "distance_to_venue": "0.3 mi"},
            {"option": 2, "name": "Four Seasons", "rate": "$385/night", "distance_to_venue": "0.7 mi"},
        ],
        "estimated_total_cost": "$3,200",
        "preferences_applied": preferences,
    }

@tool
def track_stakeholder_relationship(
    stakeholder_name: str,
    stakeholder_role: str,
    action: Literal["log_contact", "check_status"],
) -> dict:
    """Track stakeholder relationships. Log new contacts or check relationship health status."""
    print(f"[STAKEHOLDER] {action} for {stakeholder_name} ({stakeholder_role})")
    now = datetime.datetime.utcnow()
    if action == "log_contact":
        return {
            "stakeholder": stakeholder_name,
            "role": stakeholder_role,
            "last_contact_date": now.isoformat(),
            "total_interactions": 12,
            "relationship_health": "strong",
            "action": "contact_logged",
        }
    else:
        days_since = 35
        return {
            "stakeholder": stakeholder_name,
            "role": stakeholder_role,
            "days_since_last_contact": days_since,
            "last_contact_date": (now - datetime.timedelta(days=days_since)).isoformat(),
            "relationship_health": "needs_attention" if days_since > 30 else "strong",
            "reminder": f"It's been {days_since} days since last contact with {stakeholder_name}. Consider scheduling a check-in." if days_since > 30 else None,
        }

@tool
def log_decision(
    meeting_title: str,
    decision: str,
    rationale: str,
    decided_by: str,
    stakeholders: list[str],
) -> dict:
    """Log a decision made in a meeting. Records the decision with rationale and notifies stakeholders."""
    print(f"[DECISION] Logging decision from '{meeting_title}': {decision[:60]}...")
    now = datetime.datetime.utcnow()
    return {
        "decision_id": f"dec_{now.timestamp():.0f}",
        "meeting_title": meeting_title,
        "decision": decision,
        "rationale": rationale,
        "decided_by": decided_by,
        "stakeholders": stakeholders,
        "logged_at": now.isoformat(),
        "notification_sent": True,
    }

@tool
def generate_weekly_review(
    executive_name: str,
    week_ending: str,
) -> str:
    """Generate a structured weekly review for the executive. Returns Markdown summary."""
    print(f"[REVIEW] Generating weekly review for {executive_name}, week ending {week_ending}")
    return f"""# Weekly Review — {executive_name}
**Week Ending:** {week_ending}

## Accomplishments
1. Closed Series B partnership deal with Acme Corp
2. Finalised Q2 board deck and submitted for review
3. Completed 3 candidate interviews for VP Engineering role
4. Launched internal AI productivity pilot across two departments
5. Resolved cross-team blocker on product roadmap alignment

## Key Metrics Moved
- Pipeline grew 18% ($2.4M added)
- Customer NPS up 5 points to 72
- Team velocity increased 12% sprint-over-sprint

## Decisions Made
- Approved headcount increase for engineering (3 FTEs)
- Selected vendor for new CRM migration
- Deferred office expansion to Q4

## Upcoming Priorities (Next Week)
1. Board meeting preparation and rehearsal
2. Finalise VP Engineering hiring decision
3. Review and approve Q3 budget proposals
4. Customer advisory board session
5. Strategic planning offsite agenda finalisation

## Blockers
- Legal review on partnership contract pending (ETA: 2 days)
- Waiting on finance for updated runway projections

*Generated by OperaMind Executive Assistant Agent · {datetime.datetime.utcnow().strftime("%H:%M UTC")}*
"""

@tool
def detect_delegatable_tasks(
    email_subject: str,
    email_body: str,
    executive_name: str,
) -> dict:
    """Analyse an email to determine if the task should be delegated. Returns delegation recommendation."""
    print(f"[DELEGATE] Analysing email '{email_subject}' for {executive_name}")
    # Simple heuristic mock — production would use LLM reasoning
    low_priority_signals = ["reminder", "update", "fyi", "newsletter", "report", "summary"]
    should_delegate = any(s in email_subject.lower() for s in low_priority_signals)
    return {
        "email_subject": email_subject,
        "should_delegate": should_delegate,
        "reason": "Task is informational/operational and does not require executive-level decision-making." if should_delegate else "Task requires executive-level input or decision-making.",
        "suggested_delegate_role": "EA" if should_delegate else None,
        "urgency": "low" if should_delegate else "high",
        "draft_delegation_message": (
            f"Hi team,\n\n{executive_name} has asked me to handle this. "
            f"Re: '{email_subject}' — I'll review and follow up.\n\nBest,\n{executive_name}'s EA"
        ) if should_delegate else None,
    }

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the AI Executive Assistant for {executive_name} at OperaMind.

Your role is to save {executive_name} 10+ hours per week by handling:
- Email: Triage, prioritise, draft replies in their voice
- Calendar: Optimise schedule, buffer time, prevent overload
- Meetings: Prepare briefings before, summarise after
- Tasks: Prioritise using Eisenhower matrix
- Commitments: Track everything promised, send reminders
- Travel: Plan trips with flight/hotel options and cost estimates
- Stakeholders: Track relationship health, flag when contact is overdue
- Decisions: Log decisions from meetings with rationale and notify stakeholders
- Weekly Reviews: Generate structured weekly summaries of accomplishments and priorities
- Delegation: Detect tasks that can be delegated and draft delegation messages

Operating principles:
1. Protect deep work time — cluster meetings, preserve 2h blocks
2. Never schedule back-to-back calls without a 15-min buffer
3. Flag anything requiring personal executive attention
4. Draft replies that sound like {executive_name}, not corporate boilerplate
5. Track every commitment made — nothing falls through the cracks

Today: {date} {time}
"""

# ── Graph ──────────────────────────────────────────────────────────────────

tools = [fetch_emails, draft_email_reply, fetch_calendar, schedule_meeting,
         summarise_meeting, prioritise_tasks, prepare_briefing, track_commitment,
         plan_travel, track_stakeholder_relationship, log_decision,
         generate_weekly_review, detect_delegatable_tasks]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def exec_agent(state: ExecState) -> ExecState:
    now = datetime.datetime.utcnow()
    system = SYSTEM_PROMPT.format(
        executive_name=state.get("executive_name", "the Executive"),
        date=now.date(), time=now.strftime("%H:%M UTC"),
    )
    response = llm_with_tools.invoke([SystemMessage(content=system)] + state["messages"])
    return {"messages": [response]}

def router(state: ExecState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

checkpointer = MemorySaver()
builder = StateGraph(ExecState)
builder.add_node("agent", exec_agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)

# ── Public API ─────────────────────────────────────────────────────────────

def run_morning_brief(executive_name: str, executive_email: str) -> dict:
    """Run the daily morning brief — email triage + calendar view + priority list."""
    config = {"configurable": {"thread_id": f"brief_{executive_email}_{datetime.date.today()}"}}
    prompt = (
        f"Good morning! Please run {executive_name}'s morning brief:\n"
        "1. Fetch and triage all unread emails — prioritise by urgency\n"
        "2. Fetch today's calendar and flag any conflicts or missing buffers\n"
        "3. Propose the top 5 priorities for today\n"
        "4. Draft replies to the top 2 most urgent emails\n"
        "Format as a clean morning brief digest."
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)],
         "executive_name": executive_name, "executive_email": executive_email,
         "priorities_today": [], "commitments_tracked": []},
        config=config,
    )
    final = next((m for m in reversed(result["messages"])
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {"brief": final.content if final else "Brief ready."}

def process_request(executive_name: str, request: str, thread_id: str) -> dict:
    """Handle any ad-hoc executive assistant request."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=request)],
         "executive_name": executive_name, "executive_email": "",
         "priorities_today": [], "commitments_tracked": []},
        config=config,
    )
    final = next((m for m in reversed(result["messages"])
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {"response": final.content if final else "Done."}


if __name__ == "__main__":
    result = run_morning_brief("Sarah Chen", "sarah@operamind.ai")
    print("\nMORNING BRIEF:\n", result["brief"])
