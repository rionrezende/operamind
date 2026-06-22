import os
import logging
import datetime
from pathlib import Path

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, HtmlContent

    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

logger = logging.getLogger("operamind.email")

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "hello@operamind.ai")
FROM_NAME = os.getenv("FROM_NAME", "OperaMind")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "emails"

SUBJECTS = {
    "welcome": {
        "en": "Welcome to OperaMind, {name}! Your AI workforce is almost ready.",
        "es": "Bienvenido a OperaMind, {name}! Tu fuerza laboral IA casi esta lista.",
        "pt": "Bem-vindo ao OperaMind, {name}! Sua equipe de IA esta quase pronta.",
        "fr": "Bienvenue chez OperaMind, {name} ! Votre main-d'oeuvre IA est presque prete.",
    },
    "scanner_results": {
        "en": "{name}, we found {opportunities_count} automation opportunities",
        "es": "{name}, encontramos {opportunities_count} oportunidades de automatizacion",
        "pt": "{name}, encontramos {opportunities_count} oportunidades de automacao",
        "fr": "{name}, nous avons trouve {opportunities_count} opportunites d'automatisation",
    },
    "nurture_1": {
        "en": "3 tasks your {industry} team should never do manually",
        "es": "3 tareas que tu equipo de {industry} nunca deberia hacer manualmente",
        "pt": "3 tarefas que sua equipe de {industry} nunca deveria fazer manualmente",
        "fr": "3 taches que votre equipe {industry} ne devrait jamais faire manuellement",
    },
    "nurture_2": {
        "en": "How Brightline Commerce saved 32 hours/week (real numbers)",
        "es": "Como Brightline Commerce ahorro 32 horas/semana (numeros reales)",
        "pt": "Como a Brightline Commerce economizou 32 horas/semana (numeros reais)",
        "fr": "Comment Brightline Commerce a economise 32 heures/semaine (chiffres reels)",
    },
    "nurture_3": {
        "en": "Quick math: what is your team's time worth?",
        "es": "Calculo rapido: cuanto vale el tiempo de tu equipo?",
        "pt": "Conta rapida: quanto vale o tempo da sua equipe?",
        "fr": "Calcul rapide : combien vaut le temps de votre equipe ?",
    },
    "nurture_4": {
        "en": "Founding member pricing ends soon",
        "es": "Los precios de miembro fundador terminan pronto",
        "pt": "Precos de membro fundador acabam em breve",
        "fr": "Les tarifs membre fondateur se terminent bientot",
    },
    "nurture_5": {
        "en": "Quick question, {name}",
        "es": "Pregunta rapida, {name}",
        "pt": "Pergunta rapida, {name}",
        "fr": "Petite question, {name}",
    },
    "onboarding": {
        "en": "Your OperaMind onboarding is scheduled!",
        "es": "Tu onboarding de OperaMind esta programado!",
        "pt": "Seu onboarding do OperaMind esta agendado!",
        "fr": "Votre onboarding OperaMind est programme !",
    },
}

NURTURE_TEMPLATES = {
    1: "nurture_1",
    2: "nurture_2_case_study.html",
    3: "nurture_3_roi.html",
    4: "nurture_4_urgency.html",
    5: "nurture_5_personal.html",
}

NURTURE_SUBJECT_KEYS = {
    1: "nurture_1",
    2: "nurture_2",
    3: "nurture_3",
    4: "nurture_4",
    5: "nurture_5",
}


def _load_template(filename: str) -> str:
    """Load an HTML template from the emails directory."""
    template_path = TEMPLATES_DIR / filename
    try:
        return template_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Template not found: %s", template_path)
        return ""


def send_email(to_email: str, subject: str, html_content: str) -> bool:
    """Send an email via SendGrid, or log to console if not configured."""
    if not SENDGRID_API_KEY or not SENDGRID_AVAILABLE:
        logger.info(
            "SendGrid not configured. Email logged to console.\n"
            "  To: %s\n  Subject: %s\n  Body length: %d chars",
            to_email,
            subject,
            len(html_content),
        )
        return False

    message = Mail(
        from_email=(FROM_EMAIL, FROM_NAME),
        to_emails=to_email,
        subject=subject,
        html_content=HtmlContent(html_content),
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info(
            "Email sent to %s (status %s)", to_email, response.status_code
        )
        return True
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)
        return False


def send_welcome_email(
    to_email: str, name: str, plan: str, language: str = "en"
) -> bool:
    """Send welcome email after payment."""
    html = _load_template("welcome.html")
    if not html:
        return False

    today = datetime.date.today().strftime("%B %d, %Y")
    html = (
        html.replace("{name}", name)
        .replace("{plan}", plan)
        .replace("{date}", today)
        .replace("{calendly_link}", "https://calendly.com/operamind/onboarding")
    )

    lang = language if language in SUBJECTS["welcome"] else "en"
    subject = SUBJECTS["welcome"][lang].replace("{name}", name)

    return send_email(to_email, subject, html)


def send_scanner_results(
    to_email: str,
    name: str,
    industry: str,
    opportunities: list[str],
    language: str = "en",
) -> bool:
    """Send scanner results email."""
    html = _load_template("scanner_results.html")
    if not html:
        return False

    top_3 = opportunities[:3]
    opportunities_html = "".join(
        f'<tr><td style="padding:8px 0;border-bottom:1px solid #E5E7EB;">'
        f'{i+1}. {opp}</td></tr>'
        for i, opp in enumerate(top_3)
    )

    html = (
        html.replace("{name}", name)
        .replace("{industry}", industry)
        .replace("{opportunities_count}", str(len(opportunities)))
        .replace("{top_opportunities}", opportunities_html)
        .replace("{date}", datetime.date.today().strftime("%B %d, %Y"))
    )

    lang = language if language in SUBJECTS["scanner_results"] else "en"
    subject = (
        SUBJECTS["scanner_results"][lang]
        .replace("{name}", name)
        .replace("{opportunities_count}", str(len(opportunities)))
    )

    return send_email(to_email, subject, html)


def send_nurture_email(
    to_email: str,
    name: str,
    email_number: int,
    industry: str,
    language: str = "en",
) -> bool:
    """Send one of 5 nurture sequence emails."""
    if email_number not in NURTURE_TEMPLATES:
        logger.error("Invalid nurture email number: %d", email_number)
        return False

    template_file = NURTURE_TEMPLATES[email_number]
    html = _load_template(template_file)
    if not html:
        return False

    html = (
        html.replace("{name}", name)
        .replace("{industry}", industry)
        .replace("{date}", datetime.date.today().strftime("%B %d, %Y"))
    )

    subject_key = NURTURE_SUBJECT_KEYS[email_number]
    lang = language if language in SUBJECTS[subject_key] else "en"
    subject = (
        SUBJECTS[subject_key][lang]
        .replace("{name}", name)
        .replace("{industry}", industry)
    )

    return send_email(to_email, subject, html)


def send_onboarding_email(
    to_email: str, name: str, calendly_link: str, language: str = "en"
) -> bool:
    """Send onboarding instructions with Calendly link."""
    html = _load_template("onboarding.html")
    if not html:
        # Fall back to a simple inline template
        html = (
            f"<p>Hi {name},</p>"
            f"<p>Your onboarding call is ready to be scheduled.</p>"
            f'<p><a href="{calendly_link}">Schedule your call</a></p>'
            f"<p>— The OperaMind Team</p>"
        )
    else:
        html = (
            html.replace("{name}", name)
            .replace("{calendly_link}", calendly_link)
            .replace("{date}", datetime.date.today().strftime("%B %d, %Y"))
        )

    lang = language if language in SUBJECTS["onboarding"] else "en"
    subject = SUBJECTS["onboarding"][lang]

    return send_email(to_email, subject, html)
