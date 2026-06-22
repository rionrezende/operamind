# OperaMind — Deployment Guide

## Quick Start (5 steps to go live)

### Step 1: Railway (API)
1. Go to railway.com → New Project → Deploy from GitHub
2. Connect the rionrezende/operamind repo
3. Set environment variables in Railway dashboard:
   - ANTHROPIC_API_KEY=sk-ant-...
   - OPENAI_API_KEY=sk-proj-...
   - HUBSPOT_API_KEY=pat-...
   - SENDGRID_API_KEY=SG....
   - STRIPE_SECRET_KEY=sk_live_... (get from stripe.com)
   - ADMIN_PASSWORD=your_secure_password
4. Generate domain: Settings → Networking → Generate Domain
5. Note your URL: https://your-app.up.railway.app

### Step 2: Netlify (Website)
1. Go to app.netlify.com → Add new site
2. Drag and drop these files: index.html, agents.html, translations.js
3. Or connect GitHub and set publish directory to root
4. Note your URL: https://operamind.netlify.app

### Step 3: Stripe (Payments)
1. Go to stripe.com → Create account
2. Create 3 products:
   - Starter: $197/month recurring
   - Growth: $697/month recurring + $497 one-time setup
   - Enterprise: $1,997/month recurring
3. Copy each price ID → set as Railway env vars
4. Set up webhook: Developers → Webhooks → Add endpoint
   - URL: https://your-railway-url/api/stripe/webhook
   - Events: checkout.session.completed, customer.subscription.deleted
5. Copy webhook signing secret → STRIPE_WEBHOOK_SECRET env var

### Step 4: SendGrid (Email)
1. Verify sender identity: Settings → Sender Authentication
2. Verify your email address
3. API key already configured

### Step 5: Calendly
1. Create event type: "AI Assessment" (45 minutes)
2. Update index.html Calendly URLs with your real link
3. Redeploy to Netlify

## Environment Variables Reference

| Variable | Required | Where to Get |
|---|---|---|
| ANTHROPIC_API_KEY | Yes | console.anthropic.com |
| OPENAI_API_KEY | Yes | platform.openai.com |
| HUBSPOT_API_KEY | Yes | HubSpot Settings → Private Apps |
| SENDGRID_API_KEY | Yes | SendGrid Settings → API Keys |
| STRIPE_SECRET_KEY | For payments | stripe.com/dashboard/apikeys |
| STRIPE_WEBHOOK_SECRET | For payments | Stripe Webhooks |
| ADMIN_PASSWORD | Yes | Choose a secure password |
| DATABASE_PATH | Optional | Default: operamind.db |

## Testing Checklist
- [ ] /api/health returns 200 OK
- [ ] /api/status shows all agents OK
- [ ] /api/stripe/plans returns 3 plans
- [ ] Scanner captures leads (check /admin)
- [ ] Language switcher works (EN/ES/PT/FR)
- [ ] Calendly link opens correctly
- [ ] All 7 agent imports pass
