import logging
import tempfile
from pathlib import Path

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from speech_to_text_converter import WhisperTranscriber
from secret import api_token

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")

def get_transriber_function():
    """
    Get a transcriber function using the WhisperTranscriber.

    Returns:
        callable: An asynchronous function that transcribes voice messages to text.
    """
    transcriber = WhisperTranscriber(model_name="openai/whisper-small", model_sampling_rate=16_000)
    
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
            file_path = Path(temp_dir) / 'file_name.ogg'
            await new_file.download_to_drive(file_path)

            transcription = "".join(transcriber.transcribe_from_file(file_path))
            logger.info(transcription)
            await update.message.reply_text(transcription)
    return transcriber_function


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(api_token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler((filters.VOICE | filters.STORY) & ~filters.COMMAND, get_transriber_function()))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()