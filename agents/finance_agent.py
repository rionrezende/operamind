"""
OperaMind — Finance Agent
===========================
Framework : CrewAI 0.105+ (parallel crew: extractor + classifier + reconciler + reporter)
Model     : claude-sonnet-4-6
Purpose   : Invoice processing, AP/AR management, expense classification,
            reconciliation, anomaly detection, automated financial reporting.

pip install crewai langchain-anthropic anthropic python-dotenv pydantic

ENV: ANTHROPIC_API_KEY, QUICKBOOKS_CLIENT_ID, QUICKBOOKS_CLIENT_SECRET
"""
from __future__ import annotations
import os, json, datetime
from typing import ClassVar, Literal
from dotenv import load_dotenv
from pydantic import BaseModel

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_anthropic import ChatAnthropic

load_dotenv()
llm = "anthropic/claude-sonnet-4-6"

# ── Data models ────────────────────────────────────────────────────────────

class Invoice(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    currency: str = "USD"
    due_date: str
    line_items: list[dict] = []
    raw_text: str = ""

class ExpenseRecord(BaseModel):
    id: str
    amount: float
    merchant: str
    date: str
    employee: str
    description: str = ""

# ── Tools ──────────────────────────────────────────────────────────────────

class InvoiceExtractorTool(BaseTool):
    name: str = "extract_invoice_data"
    description: str = "Extract structured data from invoice text or PDF. Returns vendor, amount, due date, line items."

    def _run(self, raw_invoice: str) -> str:
        print(f"[INVOICE] Extracting data from invoice text ({len(raw_invoice)} chars)")
        # Production: use Claude's vision API for PDF invoices
        # Simplified extraction here
        data = {
            "vendor": "Acme Supplies Ltd",
            "invoice_number": "INV-20250601",
            "amount": 4850.00,
            "currency": "USD",
            "due_date": (datetime.date.today() + datetime.timedelta(days=30)).isoformat(),
            "line_items": [
                {"description": "Cloud infrastructure", "qty": 1, "unit_price": 3000.00},
                {"description": "Software licenses", "qty": 5, "unit_price": 370.00},
            ],
            "payment_terms": "NET 30",
            "extracted_at": datetime.datetime.utcnow().isoformat(),
        }
        return json.dumps(data)

class ExpenseClassifierTool(BaseTool):
    name: str = "classify_expense"
    description: str = "Classify an expense into the correct GL account category per company policy."

    CATEGORIES: ClassVar[dict] = {
        "software": ["software", "saas", "subscription", "license", "app"],
        "travel": ["hotel", "flight", "uber", "lyft", "airbnb", "train"],
        "office_supplies": ["staples", "paper", "printer", "toner", "office depot"],
        "meals_entertainment": ["restaurant", "coffee", "lunch", "dinner", "food"],
        "cloud_infrastructure": ["aws", "gcp", "azure", "cloudflare", "digitalocean"],
        "professional_services": ["consulting", "legal", "accounting", "advisory"],
        "marketing": ["ads", "advertising", "social media", "seo", "PR"],
    }

    def _run(self, expense_json: str) -> str:
        exp = json.loads(expense_json)
        merchant = exp.get("merchant", "").lower()
        desc = exp.get("description", "").lower()
        combined = merchant + " " + desc

        category = "other"
        for cat, keywords in self.CATEGORIES.items():
            if any(kw in combined for kw in keywords):
                category = cat
                break

        # Policy check
        amount = exp.get("amount", 0)
        requires_approval = amount > 500
        policy_violation = amount > 2000 and category == "meals_entertainment"

        return json.dumps({
            "expense_id": exp.get("id"),
            "gl_account": category,
            "gl_code": f"GL-{hash(category) % 9000 + 1000}",
            "requires_approval": requires_approval,
            "policy_violation": policy_violation,
            "violation_reason": "Meals > $2,000 requires CFO approval" if policy_violation else None,
            "tax_deductible": category not in ["meals_entertainment"],
        })

class ReconciliationTool(BaseTool):
    name: str = "reconcile_accounts"
    description: str = "Reconcile bank statement transactions against bookkeeping records. Flags discrepancies."

    def _run(self, period: str, account_id: str) -> str:
        print(f"[RECONCILIATION] Running reconciliation for {account_id} — period {period}")
        # Production: pull from QuickBooks + bank API, compare
        return json.dumps({
            "period": period,
            "account": account_id,
            "bank_balance": 284_750.00,
            "book_balance": 283_920.00,
            "difference": 830.00,
            "unmatched_transactions": [
                {"date": "2025-06-05", "amount": 530.00, "description": "Wire transfer — pending", "status": "outstanding"},
                {"date": "2025-06-10", "amount": 300.00, "description": "ACH credit — unrecorded", "status": "missing_in_books"},
            ],
            "status": "needs_review",
            "reconciled_at": datetime.datetime.utcnow().isoformat(),
        })

class AnomalyDetectorTool(BaseTool):
    name: str = "detect_anomalies"
    description: str = "Detect unusual transactions, duplicate invoices, and policy violations in financial data."

    def _run(self, transactions_json: str) -> str:
        transactions = json.loads(transactions_json)
        # Production: compare against historical averages, flag outliers
        anomalies = []
        for tx in transactions:
            if tx.get("amount", 0) > 10000:
                anomalies.append({
                    "transaction": tx.get("id"),
                    "type": "large_transaction",
                    "amount": tx.get("amount"),
                    "severity": "high",
                    "action_required": "CFO review",
                })
        return json.dumps({
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "scan_date": datetime.datetime.utcnow().isoformat(),
        })

class FinancialReportTool(BaseTool):
    name: str = "generate_financial_report"
    description: str = "Generate a financial summary report (P&L, cash flow, AP aging, AR aging)."

    def _run(self, report_type: str, period: str) -> str:
        print(f"[REPORT] Generating {report_type} for {period}")
        # Production: pull live data from QuickBooks / Xero API
        if report_type == "ap_aging":
            return json.dumps({
                "report": "AP Aging Summary",
                "period": period,
                "total_outstanding": 48_200.00,
                "current": 12_000.00,
                "30_days": 20_000.00,
                "60_days": 11_200.00,
                "90_plus_days": 5_000.00,
                "vendors_at_risk": ["Vendor A ($5k — 95 days)"],
            })
        return json.dumps({"report": report_type, "period": period, "status": "generated"})

class AccountingSystemTool(BaseTool):
    name: str = "update_accounting_system"
    description: str = "Post transactions, invoices, and journal entries to QuickBooks / Xero."

    def _run(self, entry_json: str) -> str:
        entry = json.loads(entry_json)
        print(f"[QUICKBOOKS] Posting: {entry.get('type')} — ${entry.get('amount')}")
        return json.dumps({
            "posted": True,
            "entry_id": f"QBO-{datetime.datetime.utcnow().timestamp():.0f}",
            "type": entry.get("type"),
            "amount": entry.get("amount"),
            "posted_at": datetime.datetime.utcnow().isoformat(),
        })

# ── Agents ─────────────────────────────────────────────────────────────────

invoice_processor = Agent(
    role="Invoice Processing Specialist",
    goal="Extract, validate, and post all invoices accurately and within payment terms",
    backstory="Expert at reading invoices of any format and capturing 100% accurate structured data.",
    tools=[InvoiceExtractorTool(), AccountingSystemTool()],
    llm=llm, verbose=True, max_iter=3,
)

expense_auditor = Agent(
    role="Expense Auditor",
    goal="Classify all expenses correctly and flag policy violations before payment",
    backstory="Meticulous auditor who knows every GL code and every company policy. Zero tolerance for duplicates.",
    tools=[ExpenseClassifierTool(), AnomalyDetectorTool()],
    llm=llm, verbose=True, max_iter=3,
)

reconciliation_agent = Agent(
    role="Accounts Reconciliation Specialist",
    goal="Keep books perfectly reconciled with bank records, resolving all discrepancies",
    backstory="Detail-obsessed accountant who hasn't let a reconciliation difference go unresolved in 5 years.",
    tools=[ReconciliationTool(), AccountingSystemTool()],
    llm=llm, verbose=True, max_iter=3,
)

cfo_reporter = Agent(
    role="CFO Report Generator",
    goal="Produce clear, accurate financial summaries for leadership review every week",
    backstory="Financial communicator who turns raw numbers into actionable executive narratives.",
    tools=[FinancialReportTool()],
    llm=llm, verbose=True, max_iter=3,
)

# ── Public API ─────────────────────────────────────────────────────────────

def process_invoice(raw_invoice_text: str) -> dict:
    task = Task(
        description=(
            f"Process this invoice:\n{raw_invoice_text}\n\n"
            "1. Extract all data using extract_invoice_data\n"
            "2. Detect any anomalies\n"
            "3. Post to accounting system\n"
            "4. Return a processing summary."
        ),
        expected_output="Invoice processing summary with GL posting confirmation.",
        agent=invoice_processor,
    )
    crew = Crew(agents=[invoice_processor], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    return {"result": str(result), "processed_at": datetime.datetime.utcnow().isoformat()}

def run_monthly_close(period: str = None) -> dict:
    """Run full monthly accounting close: reconcile + report."""
    period = period or datetime.date.today().strftime("%Y-%m")

    reconcile_task = Task(
        description=f"Reconcile all bank accounts for period {period}. Flag all discrepancies.",
        expected_output="Reconciliation report with all discrepancies and resolutions.",
        agent=reconciliation_agent,
    )
    report_task = Task(
        description=(
            f"Generate the following reports for {period}: AP aging, AR aging. "
            "Format as an executive summary for the CFO."
        ),
        expected_output="Executive financial summary with AP/AR aging.",
        agent=cfo_reporter,
    )
    crew = Crew(
        agents=[reconciliation_agent, cfo_reporter],
        tasks=[reconcile_task, report_task],
        process=Process.sequential,
    )
    result = crew.kickoff()
    return {"period": period, "report": str(result)}

def audit_expenses(expenses: list[dict]) -> dict:
    task = Task(
        description=(
            f"Audit these {len(expenses)} expense records:\n"
            f"{json.dumps(expenses, indent=2)}\n\n"
            "Classify each expense, detect any anomalies or policy violations, "
            "and flag items requiring approval."
        ),
        expected_output="Complete expense audit report with classifications and flags.",
        agent=expense_auditor,
    )
    crew = Crew(agents=[expense_auditor], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    return {"expenses_audited": len(expenses), "report": str(result)}


if __name__ == "__main__":
    sample_invoice = """
    INVOICE
    Vendor: Acme Cloud Solutions
    Invoice #: INV-2025-0601
    Date: 2025-06-01
    Due: 2025-07-01

    Line Items:
    - AWS infrastructure (monthly): $3,200.00
    - Support contract: $800.00

    Total: $4,000.00
    Payment terms: NET 30
    """
    result = process_invoice(sample_invoice)
    print("\nFINANCE AGENT RESULT:\n", result["result"])
