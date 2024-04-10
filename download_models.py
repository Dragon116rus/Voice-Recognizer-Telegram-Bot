from config import BotConfig
from speech_to_text_converter import WhisperTranscriber


if __name__ == "__main__":
    bot_config = BotConfig.from_yaml("configs/config.yml")
    WhisperTranscriber(model_name=bot_config.model, model_sampling_rate=16_000)