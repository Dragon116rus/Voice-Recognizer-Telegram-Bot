import logging
import tempfile
from pathlib import Path
from typing import Callable
import yaml

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import BotConfig
from speech_to_text_converter import WhisperTranscriber
from secret import api_token

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def get_transcriber_function(transcriber: WhisperTranscriber)-> Callable:
    """
    Create a transcriber function using the provided WhisperTranscriber instance.

    Args:
        transcriber (WhisperTranscriber): An instance of WhisperTranscriber.

    Returns:
        callable: An asynchronous function that transcribes voice messages to text.
    """
    async def transcriber_function(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Transcribe the voice message and reply with the generated text.

        Args:
            update (telegram.Update): The incoming Telegram update.
            context (telegram.ext.ContextTypes.DEFAULT_TYPE): The context passed by the handler.

        Returns:
            None
        """
        new_file = await update.message.effective_attachment.get_file()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "file_name.ogg"
            await new_file.download_to_drive(file_path)
            logger.info("File downloaded to transcribe")
            transcription = "".join(transcriber.transcribe_from_file(file_path))
            logger.info(transcription)
            await update.message.reply_text(transcription)

    return transcriber_function

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Send a message when the command /start is issued.

    Args:
        update (telegram.Update): The incoming Telegram update.
        context (telegram.ext.ContextTypes.DEFAULT_TYPE): The context passed by the handler.

    Returns:
        None
    """
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )

def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Send a message when the command /help is issued.

    Args:
        update (telegram.Update): The incoming Telegram update.
        context (telegram.ext.ContextTypes.DEFAULT_TYPE): The context passed by the handler.

    Returns:
        None
    """
    update.message.reply_text("Hi! I'm a speach recognition bot!")

def main() -> None:
    """Start the bot."""
    # Load configuration
    bot_config = BotConfig.from_yaml("configs/config.yml")

    # Create the WhisperTranscriber instance
    transcriber = WhisperTranscriber(model_name=bot_config.model, model_sampling_rate=16_000)

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(api_token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Add message handler for voice and story messages
    application.add_handler(MessageHandler((filters.VOICE | filters.STORY) & ~filters.COMMAND, get_transcriber_function(transcriber)))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
