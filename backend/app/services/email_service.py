import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.models.report import Report
import logging

logger = logging.getLogger(__name__)

async def send_report(email: str, report: Report):
    """
    Send a report via email.
    """
    if not settings.SMTP_USER or settings.SMTP_USER == "test@example.com":
        logger.info(f"Test mode: Would send report '{report.title}' to {email}")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_USER
        msg['To'] = email
        msg['Subject'] = f"Industry Insights Report: {report.title}"

        # Create HTML content
        html = f"""
        <html>
            <body>
                <h1>{report.title}</h1>
                <h2>Summary</h2>
                <p>{report.summary}</p>
                <h2>Full Report</h2>
                {report.content}
            </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        # Send email
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_TLS:
                server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
            logger.info(f"Successfully sent report to {email}")
    except Exception as e:
        logger.error(f"Failed to send email to {email}: {e}")
        # Don't raise in test mode
        if settings.SMTP_USER != "test@example.com":
            raise
