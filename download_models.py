from speech_to_text_converter import WhisperTranscriber


if __name__ == "__main__":
    WhisperTranscriber(model_name="openai/whisper-small", model_sampling_rate=16_000)