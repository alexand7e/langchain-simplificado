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
    try:
        contexto = retriever.buscar(pergunta, k=settings.top_k)
        resposta = responder(pergunta, contexto, persona=settings.default_persona)
        await update.message.reply_text(resposta)
    except Exception as e:
        logger.error(f"Erro ao responder: {e}")
        await update.message.reply_text("Desculpe, ocorreu um erro ao processar sua pergunta.")


async def start(update: Update, context):
    await update.message.reply_text(
        "Olá! Sou o agente educacional. Pergunte algo sobre os dados disponíveis!"
    )


def rodar():
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .connect_timeout(30)
        .read_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    logger.info("Bot do Telegram iniciado")
    app.run_polling(bootstrap_retries=5)
