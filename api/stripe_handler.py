"""
Stripe payment integration for OperaMind FastAPI application.

Works in demo mode when Stripe is not configured.
Import with:
    from api.stripe_handler import stripe_router
    app.include_router(stripe_router)
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

try:
    import stripe
except ImportError:
    stripe = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

PRICE_ID_STARTER = os.getenv("STRIPE_PRICE_STARTER", "")
PRICE_ID_GROWTH = os.getenv("STRIPE_PRICE_GROWTH", "")
PRICE_ID_ENTERPRISE = os.getenv("STRIPE_PRICE_ENTERPRISE", "")

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

PLAN_PRICE_MAP = {
    "starter": PRICE_ID_STARTER,
    "growth": PRICE_ID_GROWTH,
    "enterprise": PRICE_ID_ENTERPRISE,
}

SUCCESS_URL = "https://operamind.netlify.app/success.html?session_id={CHECKOUT_SESSION_ID}"
CANCEL_URL = "https://operamind.netlify.app/cancel.html"

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
stripe_router = APIRouter(prefix="/api/stripe", tags=["stripe"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class CheckoutRequest(BaseModel):
    plan: str  # "starter" | "growth" | "enterprise"
    customer_email: str
    customer_name: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _is_demo_mode() -> bool:
    """Return True when Stripe is not properly configured."""
    return (
        stripe is None
        or not STRIPE_SECRET_KEY
        or STRIPE_SECRET_KEY.startswith("sk_live_your")
    )


# ---------------------------------------------------------------------------
# POST /api/stripe/create-checkout-session
# ---------------------------------------------------------------------------
@stripe_router.post("/create-checkout-session")
async def create_checkout_session(body: CheckoutRequest):
    if body.plan not in PLAN_PRICE_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}")

    # Demo mode — return placeholder
    if _is_demo_mode():
        logger.info("Demo mode: returning placeholder checkout URL for plan=%s", body.plan)
        return {
            "checkout_url": "https://operamind.netlify.app/#pricing",
            "mode": "demo",
            "message": "Stripe not configured yet. Set STRIPE_SECRET_KEY in .env",
        }

    price_id = PLAN_PRICE_MAP[body.plan]
    if not price_id:
        raise HTTPException(
            status_code=400,
            detail=f"Price ID not configured for plan '{body.plan}'. Set STRIPE_PRICE_{body.plan.upper()} env var.",
        )

    try:
        session = stripe.checkout.Session.create(
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
            mode="subscription",
            customer_email=body.customer_email,
            metadata={
                "plan": body.plan,
                "customer_name": body.customer_name,
            },
            line_items=[{"price": price_id, "quantity": 1}],
        )
        logger.info(
            "Checkout session created: session_id=%s plan=%s email=%s",
            session.id,
            body.plan,
            body.customer_email,
        )
        return {"checkout_url": session.url}
    except stripe.error.StripeError as e:
        logger.error("Stripe error creating checkout session: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error creating checkout session: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# POST /api/stripe/webhook
# ---------------------------------------------------------------------------
@stripe_router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Verify signature if webhook secret is configured
    if STRIPE_WEBHOOK_SECRET and stripe:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logger.warning("Webhook: invalid payload")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            logger.warning("Webhook: invalid signature")
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        # No secret configured — parse payload directly (dev/demo mode)
        import json

        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, Exception):
            raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event.get("type") if isinstance(event, dict) else event.get("type", "")
    logger.info("Webhook received: type=%s", event_type)

    # --- checkout.session.completed ---
    if event_type == "checkout.session.completed":
        session_data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event["data"]["object"]
        metadata = session_data.get("metadata", {})
        logger.info(
            "Payment completed: plan=%s customer_name=%s email=%s session_id=%s amount_total=%s timestamp=%s",
            metadata.get("plan"),
            metadata.get("customer_name"),
            session_data.get("customer_email"),
            session_data.get("id"),
            session_data.get("amount_total"),
            datetime.now(timezone.utc).isoformat(),
        )

    # --- customer.subscription.created ---
    elif event_type == "customer.subscription.created":
        sub_data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event["data"]["object"]
        logger.info(
            "Subscription created: subscription_id=%s customer=%s status=%s",
            sub_data.get("id"),
            sub_data.get("customer"),
            sub_data.get("status"),
        )

    # --- customer.subscription.deleted ---
    elif event_type == "customer.subscription.deleted":
        sub_data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event["data"]["object"]
        logger.info(
            "Subscription cancelled: subscription_id=%s customer=%s canceled_at=%s",
            sub_data.get("id"),
            sub_data.get("customer"),
            sub_data.get("canceled_at"),
        )

    # --- invoice.payment_failed ---
    elif event_type == "invoice.payment_failed":
        invoice_data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event["data"]["object"]
        logger.info(
            "Payment failed: invoice_id=%s customer=%s subscription=%s amount_due=%s",
            invoice_data.get("id"),
            invoice_data.get("customer"),
            invoice_data.get("subscription"),
            invoice_data.get("amount_due"),
        )

    else:
        logger.debug("Unhandled webhook event type: %s", event_type)

    return {"received": True}


# ---------------------------------------------------------------------------
# GET /api/stripe/plans
# ---------------------------------------------------------------------------
@stripe_router.get("/plans")
async def get_plans():
    return [
        {
            "id": "starter",
            "name": "Starter",
            "badge": "Setup fee waived",
            "price_monthly": 197,
            "setup_fee": 0,
            "features": [
                "1 AI Agent of your choice",
                "Basic automation workflows",
                "Email support (48h response)",
                "Monthly performance reports",
                "Standard integrations",
                "Up to 1,000 tasks/month",
            ],
            "cta_text": "Get Started",
            "highlighted": False,
        },
        {
            "id": "growth",
            "name": "Growth",
            "badge": "Most Popular",
            "price_monthly": 697,
            "setup_fee": 497,
            "features": [
                "3 AI Agents included",
                "Advanced automation workflows",
                "Priority support (24h response)",
                "Weekly performance reports",
                "Premium integrations",
                "Up to 10,000 tasks/month",
                "Custom agent training",
                "Dedicated success manager",
            ],
            "cta_text": "Scale Now",
            "highlighted": True,
        },
        {
            "id": "enterprise",
            "name": "Enterprise",
            "badge": "Limited spots",
            "price_monthly": 1997,
            "setup_fee": "custom",
            "features": [
                "All 7 AI Agents included",
                "Unlimited automation workflows",
                "24/7 priority support",
                "Real-time performance dashboard",
                "Custom integrations & API access",
                "Unlimited tasks/month",
                "Bespoke agent training & fine-tuning",
                "Dedicated success team",
                "SLA guarantee",
                "On-call AI strategist",
            ],
            "cta_text": "Contact Sales",
            "highlighted": False,
        },
    ]
