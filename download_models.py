from default_configs import DEFAULT_MODEL
from speech_to_text_converter import WhisperTranscriber


if __name__ == "__main__":
    WhisperTranscriber(model_name=DEFAULT_MODEL, model_sampling_rate=16_000)