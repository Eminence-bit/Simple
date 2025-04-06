import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.logger import setup_logger
from scipy.signal import medfilt
from scipy.signal.windows import hann as hann_window

logger = setup_logger("Preprocessing")

def extract_tonal_features(y, sr):
    """Extract tonal features specific to Indian classical music"""
    # Chromagram optimized for Indian classical music scales
    chroma = librosa.feature.chroma_cqt(
        y=y, 
        sr=sr,
        bins_per_octave=24,  # Double resolution for detecting microtones
        hop_length=512,
        n_chroma=12
    )
    
    # Tonal features with refined parameters
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    # MFCC with parameters adjusted for Indian classical timbre
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=sr,
        n_mfcc=20,  # Increased MFCCs for better timbre capture
        n_mels=128,
        fmin=50,
        fmax=8000
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    
    return {
        'chroma': np.mean(chroma, axis=1),
        'tonnetz': np.mean(tonnetz, axis=1),
        'mfcc': np.mean(mfcc, axis=1),
        'mfcc_delta': np.mean(mfcc_delta, axis=1)
    }

def detect_stable_segments(y, sr, min_duration=1.5):
    """Detect stable pitch segments (potential steady notes)"""
    # Create window array explicitly
    window = hann_window(2048)
    
    # Pitch detection with parameters optimized for Indian classical music
    pitches, magnitudes = librosa.piptrack(
        y=y, 
        sr=sr,
        fmin=50,
        fmax=1500,
        threshold=0.6,
        n_fft=2048,
        hop_length=512,
        window=window  # Pass the explicit window array
    )
    
    # Using a larger kernel for smoother detection
    pitch_median = medfilt(pitches.max(axis=0), kernel_size=31)
    
    # Find stable segments with refined thresholds
    stable_segments = []
    current_segment = {'start': 0, 'pitch': pitch_median[0]}
    pitch_threshold = 20
    
    for i in range(1, len(pitch_median)):
        if abs(pitch_median[i] - current_segment['pitch']) > pitch_threshold:
            duration = (i - current_segment['start']) / (sr / 512)
            if duration >= min_duration:
                stable_segments.append({
                    'start': current_segment['start'],
                    'end': i,
                    'pitch': current_segment['pitch']
                })
            current_segment = {'start': i, 'pitch': pitch_median[i]}
    
    # Add the last segment if it's long enough
    if len(pitch_median) > 0:
        duration = (len(pitch_median) - current_segment['start']) / (sr / 512)
        if duration >= min_duration:
            stable_segments.append({
                'start': current_segment['start'],
                'end': len(pitch_median),
                'pitch': current_segment['pitch']
            })
    
    return stable_segments

def preprocess_audio(file_path, save_plot_dir="outputs"):
    try:
        logger.info("Starting audio preprocessing...")
        
        # Load audio with original sampling rate
        logger.info("Loading audio file...")
        y, sr = librosa.load(file_path, sr=None)
        logger.info(f"Loaded {file_path}, Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")

        # Advanced silence removal with higher threshold
        logger.info("Removing silence...")
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        logger.info(f"Audio trimmed from {len(y)} to {len(y_trimmed)} samples")
        
        # Normalize audio
        y_trimmed = librosa.util.normalize(y_trimmed)
        logger.info("Audio normalized")

        # Create window array explicitly for STFT
        window = hann_window(2048)

        # Extract pitch and tempo using refined parameters
        logger.info("Extracting pitch and tempo...")
        pitches, magnitudes = librosa.piptrack(
            y=y_trimmed, 
            sr=sr,
            fmin=50,
            fmax=1500,
            hop_length=512,
            n_fft=2048,
            window=window  # Pass explicit window array
        )
        logger.info(f"Pitch detection complete. Shape: {pitches.shape}")
        
        tempo, beats = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        logger.info(f"Beat tracking complete. Found {len(beats)} beats")

        # Extract frequencies and confidence
        logger.info("Processing pitch data...")
        frequencies = []
        confidence = []
        for time_idx in range(pitches.shape[1]):
            pitch_idx = magnitudes[:, time_idx].argmax()
            frequencies.append(pitches[pitch_idx, time_idx])
            confidence.append(magnitudes[pitch_idx, time_idx])
        logger.info(f"Processed {len(frequencies)} pitch frames")

        # Get stable segments
        logger.info("Detecting stable segments...")
        stable_segments = detect_stable_segments(y_trimmed, sr)
        logger.info(f"Found {len(stable_segments)} stable segments")
        
        # Extract tonal features
        logger.info("Extracting tonal features...")
        tonal_features = extract_tonal_features(y_trimmed, sr)
        logger.info("Tonal feature extraction complete")

        logger.info(f"Estimated Tempo: {tempo:.2f} BPM")

        # Save visualization
        logger.info("Generating visualizations...")
        os.makedirs(save_plot_dir, exist_ok=True)
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y_trimmed, sr=sr)
        plt.title("Trimmed Audio Waveform")
        
        plt.subplot(2, 1, 2)
        # Use explicit window array for STFT
        stft = librosa.stft(y_trimmed, n_fft=2048, hop_length=512, window=window)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(stft), ref=np.max),
            y_axis='log', 
            x_axis='time',
            sr=sr
        )
        plt.title('Power Spectrogram')
        
        plt.tight_layout()
        waveform_path = os.path.join(save_plot_dir, "waveform.png")
        plt.savefig(waveform_path)
        plt.close()
        logger.info(f"Analysis plots saved at: {waveform_path}")

        if not stable_segments:
            logger.warning("No stable segments found. This might indicate the detection parameters need adjustment.")
            return None

        # Return both overall analysis and segment-wise analysis
        logger.info("Preparing final analysis results...")
        return [{
            "segment_index": idx,
            "tempo": tempo,
            "start_time": segment['start'] / sr,
            "end_time": segment['end'] / sr,
            "waveform_path": waveform_path,
            "tonal_features": tonal_features,
            "pitch_data": {
                "frequency": frequencies[segment['start']:segment['end']],
                "confidence": confidence[segment['start']:segment['end']]
            }
        } for idx, segment in enumerate(stable_segments)]

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        logger.exception("Stack trace:")
        return None