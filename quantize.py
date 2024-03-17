from pathlib import Path
import librosa
from optimum.intel.openvino.quantization import InferRequestWrapper
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from itertools import islice

from tqdm import tqdm

from datasets import load_dataset

import gc
import shutil
import nncf
import openvino as ov

CALIBRATION_DATASET_SIZE = 50

def extract_input_features(processor, sample, sampling_rate=16_000):
    data = sample["audio"]["array"]
    data = librosa.resample(data, orig_sr=sample["audio"]["sampling_rate"], target_sr=sampling_rate)

    input_features = processor(
        data,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features
    return input_features


def collect_calibration_dataset(processor, ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    calibration_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="validation", streaming=True)
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(processor, sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data

def quantize(model_path: Path, quantized_model_path: Path, processor, ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
    ov_config = {"CACHE_DIR": ""}

    if not quantized_model_path.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
            processor, ov_model, calibration_dataset_size
        )
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            subset_size=len(encoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
        )
        ov.save_model(quantized_encoder, quantized_model_path / "openvino_encoder_model.xml")
        del quantized_encoder
        del encoder_calibration_data
        gc.collect()

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            subset_size=len(decoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
        )
        ov.save_model(quantized_decoder_with_past, quantized_model_path / "openvino_decoder_with_past_model.xml")
        del quantized_decoder_with_past
        del decoder_calibration_data
        gc.collect()

        # Copy the config file and the first-step-decoder manually
        shutil.copy(model_path / "config.json", quantized_model_path / "config.json")
        shutil.copy(model_path / "openvino_decoder_model.xml", quantized_model_path / "openvino_decoder_model.xml")
        shutil.copy(model_path / "openvino_decoder_model.bin", quantized_model_path / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path, ov_config=ov_config, compile=False)
    quantized_ov_model.to('CPU')
    quantized_ov_model.compile()
    return quantized_ov_model


if __name__ == "__main__":
    model_name = "lorenzoncina/whisper-small-ru"
    model_path = Path(model_name.replace('/', '_'))
    ov_config = {"CACHE_DIR": ""}
    if not model_path.exists():
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_name, ov_config=ov_config, export=True, compile=True, load_in_8bit=False
        )
        ov_model.save_pretrained(model_path)
    else:
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_path, ov_config=ov_config, compile=True, export=False,
        )

    processor = AutoProcessor.from_pretrained(model_name)
    ov_quantized_model = quantize(
        model_path, 
        Path("quantized"), 
        processor, 
        ov_model, 
        CALIBRATION_DATASET_SIZE
        )
    