import torch
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os
from models.wav2vac2 import wav2vec2model
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transcribe_audio(audio_path, upload_dir, segment_length=30):

    model ,processor = wav2vec2model()

    try:
        audio = AudioSegment.from_wav(audio_path)
        duration = int(len(audio) / 1000)  # Convert to seconds and ensure it's an integer

        transcriptions = []
        for start in range(0, duration, segment_length):
            end = min(start + segment_length, duration)
            segment = audio[start * 1000:end * 1000]
            segment_path = os.path.join(upload_dir, f"segment_{start}_{end}.wav")
            segment.export(segment_path, format="wav")

            # Load audio using scipy instead of torchaudio to avoid TorchCodec issues
            sample_rate, waveform_data = wavfile.read(segment_path)
            
            # Convert to float32 and normalize to [-1, 1]
            if waveform_data.dtype != np.float32:
                waveform_data = waveform_data.astype(np.float32) / 32768.0
            
            # Handle stereo by taking the first channel
            if len(waveform_data.shape) > 1:
                waveform_data = waveform_data[:, 0]

            # Resample to 16kHz if needed
            resample_rate = 16000
            if sample_rate != resample_rate:
                # Simple resampling using numpy interpolation
                num_samples = int(len(waveform_data) * resample_rate / sample_rate)
                indices = np.linspace(0, len(waveform_data) - 1, num_samples)
                waveform_data = np.interp(indices, np.arange(len(waveform_data)), waveform_data)

            inputs = processor(waveform_data, return_tensors="pt", padding="longest", sampling_rate=resample_rate).to(device)

            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            transcriptions.append((start, transcription))

            os.remove(segment_path)

        return transcriptions
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return []