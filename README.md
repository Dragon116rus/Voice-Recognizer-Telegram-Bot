# Audio tools

## Overview

This repo contain 2 scripts: 
- Audio Modifier: script for change the speed and volume of the audio file
- Speech to Text Converter: script for transcribing audio into text

## Requirements

- Python 3.8 or later
- Docker (optional, for containerization)
- Internet connection (for downloading model weights)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Dragon116rus/audio_tools.git
    ```

2. Change to the project directory:

    ```bash
    cd audio_tools
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. ### Audio Modifier

    To modify the speed and volume of an audio file, run the `audio_modifier.py` script:

    ```bash
    python audio_modifier.py --input_file /path/to/your/input/audio.wav --output_file /path/to/output/audio_modified.wav --speed_factor 1.5 --volume_factor 0.8
    ```

    Options
    ```
    --input_file: Path to the input audio file.
    --output_file: Path to save the modified audio file.
    --speed_factor: Factor to modify the speed of the audio (default is 1.0).
    --volume_factor: Factor to modify the volume of the audio (default is 1.0).
    ```

    #### Docker Usage
    Alternatively, you can use Docker for containerization. Build the Docker image:

    ```bash
    docker build -t audio_tools .
    ```

    Run the Docker container with a volume for the "output" directory:
    ```bash
    docker run -v /path/on/host:/app/output --audio-modifier --input_file /path/on/container/audio.wav --output_file /app/output/audio_modified.wav --speed_factor 2.0 --volume_factor 0.5
    ```


2. ### Speech to Text Converter

    To transcribe an audio file, use the `speech_to_text_converter.py` script:

    ```bash
    python speech_to_text_converter --input_file /path/to/your/audio/file.wav
    ```

    Options
    ```
    --input_file: Path to the input audio file.    
    --model_name: Specify the Whisper model name. Default is "openai/whisper-small".
    --model_sampling_rate: Specify the sampling rate required for the model. Default is 16000.
    --output_json: Specify the path to save the transcription in JSON format. Default is "output/result.json".
    ```

    Example with options:
    ```bash
    python main.py --input_file /path/to/audio.wav --model_name "openai/whisper-medium" --model_sampling_rate 16000 --output_json /path/to/output.json
    ```

    #### Docker Usage
    Alternatively, you can use Docker for containerization. Build the Docker image:

    ```bash
    docker build -t audio_tools .
    ```

    Run the Docker container with a volume for the "output" directory:
    ```bash
    docker run -v /path/on/host:/app/output audio_tools --speech-to-text --input_file /path/on/container/audio.wav --model_name "openai/whisper-medium" --model_sampling_rate 16000 --output_json /app/output/output.json
    ```

