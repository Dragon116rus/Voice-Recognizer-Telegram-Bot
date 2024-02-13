import argparse
import json
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperTranscriber:
    """
    A class for transcribing audio using the Whisper ASR model.
    """

    def __init__(self, model_name: str = "openai/whisper-small", model_sampling_rate: int = 16000):
        """
        Initialize the WhisperTranscriber instance.

        Parameters:
            - model_name (str, optional): The name or path of the Whisper ASR model (default is "openai/whisper-small").
            - target_sr (int, optional): Target sampling rate for resampling audio (default is 16000).
        """
        self.processor, self.model = self.load_model_and_processor(model_name)
        self.model_sampling_rate = model_sampling_rate

    def load_model_and_processor(self, model_name: str):
        """
        Load the Whisper ASR model and processor.

        Parameters:
            - model_name (str): The name or path of the Whisper ASR model.

        Returns:
            Tuple: Tuple containing the processor and model instances.
        """
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.config.forced_decoder_ids = None
        return processor, model

    def process_audio_file(self, audio_path: str) -> np.ndarray:
        """
        Process the audio file by loading and resampling it.

        Parameters:
            - audio_path (str): Path to the input audio file.

        Returns:
            np.ndarray: Processed audio data.
        """
        data, sampling_rate = librosa.load(audio_path)
        data = librosa.resample(data, orig_sr=sampling_rate, target_sr=self.model_sampling_rate)
        return data

    def transcribe_audio(self, data: np.ndarray) -> str:
        """
        Transcribe the given audio data.

        Parameters:
            - data (np.ndarray): Audio data.

        Returns:
            str: Transcription result.
        """
        input_features = self.processor(data, sampling_rate=self.model_sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription

    def transcribe_from_file(self, audio_path: str) -> str:
        """
        Transcribe audio from the specified file.

        Parameters:
            - audio_path (str): Path to the input audio file.

        Returns:
            str: Transcription result.
        """
        data = self.process_audio_file(audio_path)
        transcription = self.transcribe_audio(data)
        return transcription
    
    
def save_transcription_to_json(transcription_result: str, output_json_file: str):
    """
    Save the transcription result into a JSON file.

    Parameters:
        - output_json_file (str): Path to the output JSON file.
    """
    if transcription_result is not None:
        transcription_data = {"transcription": transcription_result}
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(transcription_data, json_file, ensure_ascii=False)
    else:
        print("No transcription result to save. Run transcribe_audio() first.")

def main(args):
    """
    Main function for transcribing audio using the WhisperTranscriber class.

    Parameters:
        - args (argparse.Namespace): Command-line arguments.
    """
    transcriber = WhisperTranscriber(model_name=args.model_name, model_sampling_rate=args.model_sampling_rate)
    transcription = transcriber.transcribe_from_file(args.input_file)
    save_transcription_to_json(transcription, args.output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper ASR model.")
    parser.add_argument("--input_file", type=str, help="Path to the input audio file.", required=True)
    parser.add_argument("--model_name", type=str, default="openai/whisper-small", help="Whisper model name.")
    parser.add_argument("--model_sampling_rate", type=int, default=16000, help="Sampling rate required for model (default is 16000).")
    parser.add_argument("--output_json", type=str, default="result.json", help="Path to the output JSON file for saving the transcription.")

    args = parser.parse_args()
    main(args)