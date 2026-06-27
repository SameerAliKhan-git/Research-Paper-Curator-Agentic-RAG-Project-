import logging
import time
from typing import Optional, List
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.schemas.api.ask import AskRequest, AskResponse
from src.database import get_db_session
from src.models.researcher import DailyBriefing, ResearcherInterest
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.config import GraphConfig

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 5  # Minimum seconds between requests per user


class TelegramBot:
    """Telegram bot connected to the advanced agentic RAG pipeline, supporting subscriptions and briefings."""

    def __init__(
        self,
        bot_token: str,
        opensearch_client,
        embeddings_client,
        ollama_client,
        cache_client=None,
        langfuse_tracer=None,
        model: str = "llama3.2:1b",
    ):
        """Initialize bot with required services."""
        self.bot_token = bot_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache = cache_client
        self.langfuse = langfuse_tracer
        self.model = model
        self.application: Optional[Application] = None
        self._user_timestamps: dict = {}  # Simple rate limiting per user

        # Initialize the advanced agentic RAG service
        self.agentic_rag = AgenticRAGService(
            opensearch_client=opensearch_client,
            ollama_client=ollama_client,
            embeddings_client=embeddings_client,
            langfuse_tracer=langfuse_tracer,
            graph_config=GraphConfig(model=model)
        )

    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is within rate limit. Returns True if allowed."""
        now = time.time()
        last_request = self._user_timestamps.get(user_id, 0)
        if now - last_request < RATE_LIMIT_SECONDS:
            return False
        self._user_timestamps[user_id] = now
        return True

    async def start(self) -> None:
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        self.application = Application.builder().token(self.bot_token).build()

        # Register handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("search", self._search_command))
        self.application.add_handler(CommandHandler("briefing", self._briefing_command))
        self.application.add_handler(CommandHandler("subscribe", self._subscribe_command))
        self.application.add_handler(CommandHandler("unsubscribe", self._unsubscribe_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_question))

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Telegram bot started successfully")

    async def stop(self) -> None:
        """Stop the bot."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

    async def notify_new_paper(self, title: str, arxiv_id: str, summary: str, chat_id: str) -> None:
        """Send a notification about a new paper to a specific user chat ID."""
        if not self.application or not self.application.bot:
            return
        
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        message = (
            f"🔔 *New Paper Ingested!*\n\n"
            f"*Title:* {title}\n"
            f"*arXiv URL:* {arxiv_url}\n\n"
            f"*Quick Summary:* {summary[:300]}..."
        )
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to arXiv Paper Curator!\n\n"
            "Ask me questions about CS papers and I'll provide answers with sources using my Agentic RAG engine.\n\n"
            "Commands:\n"
            "/help - Show this help\n"
            "/search <keywords> - Search papers\n"
            "/briefing - Get the latest daily brief of relevant papers\n"
            "/subscribe <keyword> - Subscribe to research interest keywords\n"
            "/unsubscribe <keyword> - Remove subscription keywords"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Send me any question about computer science research papers.\n\n"
            "Examples:\n"
            "- What are transformer architectures?\n"
            "- Explain attention mechanisms\n\n"
            "Other features:\n"
            "- Subscribe to keywords to monitor new releases: `/subscribe machine learning`\n"
            "- View latest briefing details: `/briefing`"
        )

    async def _search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search command."""
        if not context.args:
            await update.message.reply_text("Usage: /search <keywords>\nExample: /search neural networks")
            return

        if not self._check_rate_limit(update.effective_user.id):
            await update.message.reply_text("Please wait a few seconds before making another request.")
            return

        query = " ".join(context.args)
        await update.message.chat.send_action("typing")

        try:
            query_embedding = await self.embeddings.embed_query(query)
            results = await self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=True,
            )

            hits = results.get("hits", [])
            if not hits:
                await update.message.reply_text("No papers found. Try different keywords.")
                return

            seen_ids = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            response = f"Found {len(unique_papers)} papers:\n\n"
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                url = f"https://arxiv.org/abs/{arxiv_id}"
                response += f"{idx}. {title}\n{url}\n\n"

            await update.message.reply_text(response, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await update.message.reply_text("Search failed. Please try again later.")

    async def _briefing_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /briefing command."""
        try:
            with get_db_session() as session:
                briefings = session.query(DailyBriefing).order_by(DailyBriefing.created_at.desc()).limit(5).all()
                if not briefings:
                    await update.message.reply_text("No daily briefings available yet. The Airflow curation tasks generate these.")
                    return

                response = "📚 *Latest Daily Briefings:*\n\n"
                for idx, b in enumerate(briefings, 1):
                    arxiv_url = f"https://arxiv.org/abs/{b.arxiv_id}"
                    response += f"{idx}. *{b.title}* (Score: {b.score:.2f})\n"
                    response += f"Summary: {b.summary[:250]}...\n{arxiv_url}\n\n"

                await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Briefing command failed: {e}")
            await update.message.reply_text("Failed to load briefings. Please try again later.")

    async def _subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /subscribe command."""
        if not context.args:
            await update.message.reply_text("Usage: /subscribe <keyword>\nExample: /subscribe reinforcement learning")
            return

        keyword = " ".join(context.args).strip().lower()
        try:
            with get_db_session() as session:
                existing = session.query(ResearcherInterest).filter(ResearcherInterest.keyword == keyword).first()
                if existing:
                    await update.message.reply_text(f"You are already subscribed to: '{keyword}'")
                    return

                new_interest = ResearcherInterest(keyword=keyword)
                session.add(new_interest)
                session.commit()
                await update.message.reply_text(f"Successfully subscribed to keyword: '{keyword}'! You will receive alerts when matching papers are ingested.")
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            await update.message.reply_text("Subscription failed. Please try again.")

    async def _unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unsubscribe command."""
        if not context.args:
            await update.message.reply_text("Usage: /unsubscribe <keyword>\nExample: /unsubscribe reinforcement learning")
            return

        keyword = " ".join(context.args).strip().lower()
        try:
            with get_db_session() as session:
                interest = session.query(ResearcherInterest).filter(ResearcherInterest.keyword == keyword).first()
                if not interest:
                    await update.message.reply_text(f"Subscription keyword not found: '{keyword}'")
                    return

                session.delete(interest)
                session.commit()
                await update.message.reply_text(f"Successfully unsubscribed from keyword: '{keyword}'")
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            await update.message.reply_text("Unsubscription failed. Please try again.")

    async def _handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user questions by routing them through the agentic RAG service."""
        if not self._check_rate_limit(update.effective_user.id):
            await update.message.reply_text("Please wait a few seconds before making another request.")
            return

        query = update.message.text
        await update.message.chat.send_action("typing")

        try:
            # Execute RAG query using the advanced AgenticRAGService
            result = await self.agentic_rag.ask(query=query, user_id=f"tg_{update.effective_user.id}")
            
            answer = result.get("answer", "I couldn't generate an answer.")
            sources = result.get("sources", [])

            # Format answer and sources
            message = f"*Answer:*\n{answer}\n"
            if sources:
                message += "\n*Sources:*\n"
                for idx, src in enumerate(sources[:5], 1):
                    # Format standard arXiv or web links
                    if "arxiv.org" in src:
                        arxiv_id = src.split("/")[-1].replace(".pdf", "")
                        message += f"{idx}. https://arxiv.org/abs/{arxiv_id}\n"
                    else:
                        message += f"{idx}. {src}\n"

            # Send message to user
            try:
                await update.message.reply_text(message, parse_mode="Markdown", disable_web_page_preview=True)
            except Exception:
                await update.message.reply_text(message, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await update.message.reply_text("An error occurred while processing your question. Please try again.")
