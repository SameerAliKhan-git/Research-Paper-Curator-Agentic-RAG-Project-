import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any

from src.config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service to send HTML daily briefings to subscribed researchers using SMTP."""

    def __init__(self):
        self.settings = get_settings()

    def _build_html_body(self, briefing_data: List[Dict[str, Any]]) -> str:
        """Create HTML email markup for research paper briefing digests."""
        rows_html = ""
        for idx, item in enumerate(briefing_data, 1):
            title = item.get("title", "Untitled Paper")
            arxiv_id = item.get("arxiv_id", "")
            summary = item.get("summary", "")
            score = item.get("score", 0.0)
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            
            rows_html += f"""
            <div style="margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid #eeeeee;">
                <h3 style="margin: 0 0 8px 0; color: #1a0dab;">
                    {idx}. <a href="{arxiv_url}" style="text-decoration: none;">{title}</a>
                </h3>
                <div style="font-size: 13px; color: #666666; margin-bottom: 8px;">
                    <strong>arXiv ID:</strong> {arxiv_id} | <strong>Relevance Score:</strong> {score:.2f}
                </div>
                <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #333333;">
                    {summary}
                </p>
            </div>
            """

        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f6f9fc; color: #333333;">
                <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 15px; margin-top: 0;">
                        arXiv Paper Curator - Daily Research Briefing 📚
                    </h2>
                    <p style="font-size: 15px; color: #555555; line-height: 1.5;">
                        Here are today's top curated research papers matching your interests:
                    </p>
                    <div style="margin-top: 25px;">
                        {rows_html}
                    </div>
                    <footer style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #eeeeee; font-size: 12px; color: #999999; text-align: center;">
                        This briefing was compiled automatically by the Agentic RAG engine.<br>
                        To update your subscription preferences, visit the web dashboard.
                    </footer>
                </div>
            </body>
        </html>
        """
        return html

    async def send_daily_briefing(self, to_email: str, briefing_data: List[Dict[str, Any]]) -> bool:
        """Send daily briefings to researcher's email address."""
        if not self.settings.email.enabled:
            logger.info("Email service is disabled. Mocking email transmission instead.")
            logger.debug(f"Mock briefing email content sent to: {to_email} with {len(briefing_data)} items")
            return True

        if not self.settings.email.smtp_user or not self.settings.email.smtp_password:
            logger.warning("Email service is enabled but SMTP credentials are not configured.")
            return False

        if not briefing_data:
            logger.info("No briefing data to send.")
            return True

        try:
            # Build MIME message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "arXiv Paper Curator: Daily Research Briefing 📚"
            msg["From"] = self.settings.email.from_address
            msg["To"] = to_email

            html_content = self._build_html_body(briefing_data)
            msg.attach(MIMEText(html_content, "html"))

            # SMTP Connection
            logger.info(f"Connecting to SMTP server {self.settings.email.smtp_host}:{self.settings.email.smtp_port}...")
            server = smtplib.SMTP(self.settings.email.smtp_host, self.settings.email.smtp_port)
            server.starttls()
            server.login(self.settings.email.smtp_user, self.settings.email.smtp_password)
            
            # Send email
            server.sendmail(self.settings.email.from_address, to_email, msg.as_string())
            server.quit()
            
            logger.info(f"Successfully sent daily briefing email to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send briefing email to {to_email}: {e}")
            return False
