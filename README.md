# Telegram Speech-to-Text Bot

This Python script implements a Telegram bot that transcribes voice messages to text using the Whisper ASR (Automatic Speech Recognition) model provided by OpenAI.

## Prerequisites

- Python 3.7 or later
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) library
- [whisper-asr](https://github.com/openai/openai-whisper-asr) library
- Other dependencies mentioned in the script

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a Telegram bot using BotFather on Telegram and obtain the API token.

3. Create a secret.py file with the following content, replacing 'YOUR_TELEGRAM_API_TOKEN' with the actual token:

```python
api_token = 'YOUR_TELEGRAM_API_TOKEN'
```

## Usage

Run the script by executing the following commands:

```python
python download_models.py
python main.py
```
or use Dockerfile to build image.

The bot will be active and respond to commands and voice messages on Telegram.

Commands
`/start`: Start the bot and receive a welcome message.
`/help`: Get help and information about using the bot.
