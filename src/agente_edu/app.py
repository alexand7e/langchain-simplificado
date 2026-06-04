import logging
import sys

from agente_edu.config import settings


def configurar_log():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main():
    configurar_log()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando agente educacional")
    logger.info(f"Interface selecionada: {settings.interface}")

    if settings.interface == "telegram":
        from agente_edu.interfaces.telegram_bot import rodar
        rodar()
    elif settings.interface == "web":
        from agente_edu.interfaces.web_api import rodar
        rodar()
    else:
        logger.error(f"Interface desconhecida: {settings.interface}")
        sys.exit(1)


if __name__ == "__main__":
    main()
