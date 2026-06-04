import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from agente_edu.chains.qa_chain import responder
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()


async def on_message(update: Update, context):
    pergunta = update.message.text
    logger.info(f"Pergunta recebida: {pergunta}")
    contexto = retriever.buscar(pergunta, k=settings.top_k)
    resposta = responder(pergunta, contexto, persona=settings.default_persona)
    await update.message.reply_text(resposta)


async def start(update: Update, context):
    await update.message.reply_text(
        "Olá! Sou o agente educacional. Pergunte algo sobre os dados disponíveis!"
    )


def rodar():
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    logger.info("Bot do Telegram iniciado")
    app.run_polling()
