"""
OperaMind — Recruiting Agent
==============================
Framework : CrewAI 0.105+ (role-based multi-agent crew)
Model     : claude-sonnet-4-6
Purpose   : Screens resumes, conducts async interviews, schedules interviews,
            creates scorecards, sends status updates.

pip install crewai crewai-tools langchain-anthropic anthropic python-dotenv pydantic

ENV: ANTHROPIC_API_KEY, CALENDLY_API_KEY
"""
from __future__ import annotations
import os, json, datetime
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = "anthropic/claude-sonnet-4-6"

# ── Data models ────────────────────────────────────────────────────────────

class Candidate(BaseModel):
    name: str
    email: str
    resume_text: str
    role_applied: str
    years_experience: int = 0
    linkedin_url: str = ""

class JobRequirements(BaseModel):
    role: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    min_years_experience: int
    team: str
    hiring_manager: str

class Scorecard(BaseModel):
    candidate_name: str
    role: str
    total_score: float          # 0-100
    skill_match: float
    experience_match: float
    culture_signals: float
    recommendation: str         # 'Strong Yes' | 'Yes' | 'Maybe' | 'No'
    strengths: list[str]
    concerns: list[str]
    interview_notes: str

# ── Tools (CrewAI BaseTool pattern) ────────────────────────────────────────

class ResumeScreenerTool(BaseTool):
    name: str = "screen_resume"
    description: str = "Screen a candidate resume against job requirements. Returns a preliminary score and key findings."

    def _run(self, candidate_json: str, requirements_json: str) -> str:
        candidate = json.loads(candidate_json)
        reqs = json.loads(requirements_json)
        # In production: use embeddings / skills parser
        resume = candidate.get("resume_text", "").lower()
        must_have = reqs.get("must_have_skills", [])
        found = [s for s in must_have if s.lower() in resume]
        score = (len(found) / max(len(must_have), 1)) * 100
        return json.dumps({
            "preliminary_score": round(score, 1),
            "skills_found": found,
            "skills_missing": [s for s in must_have if s not in found],
            "years_experience": candidate.get("years_experience", 0),
            "min_required": reqs.get("min_years_experience", 0),
            "experience_met": candidate.get("years_experience", 0) >= reqs.get("min_years_experience", 0),
        })

class AsyncInterviewTool(BaseTool):
    name: str = "conduct_async_interview"
    description: str = "Send structured async interview questions to a candidate and record responses."

    def _run(self, candidate_email: str, role: str, questions: str) -> str:
        print(f"[ASYNC INTERVIEW] Sending questions to {candidate_email} for {role} role")
        # Production: integrate with Spark Hire / HireVue / custom form
        return json.dumps({
            "sent": True,
            "candidate_email": candidate_email,
            "questions_sent": json.loads(questions) if questions.startswith("[") else [questions],
            "response_deadline": (datetime.date.today() + datetime.timedelta(days=3)).isoformat(),
            "interview_link": f"https://interview.operamind.ai/{candidate_email.split('@')[0]}",
        })

class ScheduleInterviewTool(BaseTool):
    name: str = "schedule_interview"
    description: str = "Schedule a live interview via Calendly. Returns meeting link and time."

    def _run(self, candidate_email: str, candidate_name: str, interview_type: str) -> str:
        print(f"[CALENDAR] Scheduling {interview_type} for {candidate_name}")
        return json.dumps({
            "success": True,
            "meeting_link": f"https://calendly.com/operamind-hiring/{interview_type.lower().replace(' ','-')}?email={candidate_email}",
            "type": interview_type,
            "instructions_sent": True,
        })

class ScorecardGeneratorTool(BaseTool):
    name: str = "generate_scorecard"
    description: str = "Generate a structured candidate scorecard for hiring manager review."

    def _run(self, scorecard_json: str) -> str:
        data = json.loads(scorecard_json)
        # Format as Markdown
        md = f"""
# Candidate Scorecard: {data.get('candidate_name')}
**Role:** {data.get('role')} | **Score:** {data.get('total_score')}/100

## Recommendation: {data.get('recommendation')}

| Dimension         | Score |
|-------------------|-------|
| Skills Match      | {data.get('skill_match')}/40 |
| Experience        | {data.get('experience_match')}/30 |
| Culture Signals   | {data.get('culture_signals')}/30 |

## Strengths
{chr(10).join('- ' + s for s in data.get('strengths', []))}

## Concerns
{chr(10).join('- ' + c for c in data.get('concerns', []))}

## Interview Notes
{data.get('interview_notes', '')}

*Generated by OperaMind Recruiting Agent · {datetime.date.today()}*
"""
        print(f"[SCORECARD] Generated for {data.get('candidate_name')}")
        return md

class CandidateCommunicationTool(BaseTool):
    name: str = "send_candidate_email"
    description: str = "Send status update emails to candidates (shortlist, rejection, interview invite)."

    def _run(self, to_email: str, candidate_name: str, email_type: str, custom_note: str = "") -> str:
        templates = {
            "shortlist": f"Congratulations {candidate_name}! We'd like to move forward with your application.",
            "rejection": f"Thank you {candidate_name} for your interest. After careful review, we won't be moving forward at this time.",
            "interview_invite": f"Hi {candidate_name}! We'd like to schedule an interview. Please use the link below.",
            "offer": f"We're thrilled to extend an offer to join our team, {candidate_name}!",
        }
        body = templates.get(email_type, custom_note)
        print(f"[EMAIL] {email_type} → {candidate_name} <{to_email}>")
        return json.dumps({"sent": True, "type": email_type, "to": to_email})

# ── CrewAI Agents ──────────────────────────────────────────────────────────

resume_screener = Agent(
    role="Resume Screening Specialist",
    goal="Screen all resumes against job requirements accurately and objectively",
    backstory=(
        "You are an expert at parsing resumes and matching candidates to role requirements. "
        "You are objective, thorough, and fast. You screen 100% of applicants — no resume goes unreviewed."
    ),
    tools=[ResumeScreenerTool()],
    llm=llm,
    verbose=True,
    max_iter=3,
)

interview_coordinator = Agent(
    role="Interview Coordinator",
    goal="Conduct structured async interviews and schedule live interviews efficiently",
    backstory=(
        "You run smooth, professional interview processes. You prepare sharp, role-relevant questions, "
        "send them to candidates, and coordinate scheduling with zero back-and-forth."
    ),
    tools=[AsyncInterviewTool(), ScheduleInterviewTool()],
    llm=llm,
    verbose=True,
    max_iter=3,
)

evaluation_agent = Agent(
    role="Candidate Evaluation Specialist",
    goal="Generate objective, data-driven candidate scorecards for hiring manager review",
    backstory=(
        "You synthesize all available candidate data — resume, interview responses, experience — "
        "into a structured scorecard that helps hiring managers make confident, fast decisions."
    ),
    tools=[ScorecardGeneratorTool(), CandidateCommunicationTool()],
    llm=llm,
    verbose=True,
    max_iter=3,
)

# ── Public API ─────────────────────────────────────────────────────────────

def process_candidate(candidate: Candidate, job: JobRequirements) -> dict:
    """
    Full recruiting pipeline for a single candidate:
    Screen → Interview → Scorecard → Communicate
    """

    screen_task = Task(
        description=(
            f"Screen this candidate for the {job.role} role:\n"
            f"Candidate: {candidate.model_dump_json()}\n"
            f"Requirements: {job.model_dump_json()}\n\n"
            f"Use the screen_resume tool. Determine if they meet the must-have criteria."
        ),
        expected_output="A preliminary score, list of matched skills, and a pass/fail decision.",
        agent=resume_screener,
    )

    interview_task = Task(
        description=(
            f"If the candidate passed screening, conduct an async interview for {job.role}.\n"
            f"Candidate email: {candidate.email}\n"
            f"Generate 4 targeted questions based on the role requirements and the resume screening results."
        ),
        expected_output="Confirmation that async interview was sent and interview link generated.",
        agent=interview_coordinator,
        context=[screen_task],
    )

    scorecard_task = Task(
        description=(
            f"Generate a structured scorecard for {candidate.name} applying for {job.role}.\n"
            f"Synthesize the screening results and interview. Score each dimension (skill_match/40, "
            f"experience_match/30, culture_signals/30). Give a final recommendation.\n"
            f"Then send the appropriate status email to {candidate.email}."
        ),
        expected_output="A complete Markdown scorecard + confirmation that candidate was emailed.",
        agent=evaluation_agent,
        context=[screen_task, interview_task],
    )

    crew = Crew(
        agents=[resume_screener, interview_coordinator, evaluation_agent],
        tasks=[screen_task, interview_task, scorecard_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return {
        "candidate": candidate.name,
        "role": job.role,
        "scorecard": str(result),
        "processed_at": datetime.datetime.utcnow().isoformat(),
    }


def batch_screen(candidates: list[Candidate], job: JobRequirements) -> list[dict]:
    """Screen multiple candidates for the same role in sequence."""
    return [process_candidate(c, job) for c in candidates]


if __name__ == "__main__":
    candidate = Candidate(
        name="Maria Santos",
        email="maria@example.com",
        resume_text=(
            "5 years experience in Python, LangChain, AI agents. "
            "Built production RAG systems. Strong communication skills. "
            "Previous role: AI Engineer at Acme Corp."
        ),
        role_applied="AI Engineer",
        years_experience=5,
    )
    job = JobRequirements(
        role="AI Engineer",
        must_have_skills=["Python", "LangChain", "AI agents"],
        nice_to_have_skills=["CrewAI", "RAG", "LangGraph"],
        min_years_experience=3,
        team="Engineering",
        hiring_manager="CTO",
    )
    result = process_candidate(candidate, job)
    print("\nRECRUITING RESULT:\n", result["scorecard"])
