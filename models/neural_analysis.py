import torch
import torchaudio
import torchaudio.transforms as T
import openl3
import numpy as np
from utils.logger import setup_logger
import librosa

logger = setup_logger("NeuralAnalysis")

class AudioEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize OpenL3 model for music embeddings
        self.openl3_model = openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=512
        )
        
        # Initialize audio transforms
        self.melspec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(self.device)
        
        self.mfcc = T.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                'n_fft': 2048,
                'n_mels': 128,
                'hop_length': 512
            }
        ).to(self.device)
        
        logger.info("Neural models initialized successfully")

    def extract_audio_features(self, audio, sr):
        """Extract audio features using torchaudio transforms"""
        try:
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            waveform = torch.tensor(audio).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                # Extract mel spectrogram and MFCC features
                mel_spec = self.melspec(waveform)
                mfcc = self.mfcc(waveform)
                
                # Average across time dimension
                mel_features = torch.mean(mel_spec, dim=2)
                mfcc_features = torch.mean(mfcc, dim=2)
                
                # Combine features
                combined = torch.cat([mel_features, mfcc_features], dim=1)
                
            return combined.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None

    def extract_openl3_embeddings(self, audio, sr):
        """Extract OpenL3 embeddings"""
        try:
            emb, _ = openl3.get_audio_embedding(
                audio, sr, model=self.openl3_model,
                hop_size=0.5  # 0.5 second hop size
            )
            return emb
            
        except Exception as e:
            logger.error(f"Error extracting OpenL3 embeddings: {e}")
            return None

    def get_combined_embeddings(self, audio, sr):
        """Get combined embeddings from both models"""
        audio_features = self.extract_audio_features(audio, sr)
        openl3_emb = self.extract_openl3_embeddings(audio, sr)
        
        if audio_features is not None and openl3_emb is not None:
            # Average OpenL3 embeddings across time
            openl3_avg = np.mean(openl3_emb, axis=0)
            
            # Concatenate embeddings
            combined = np.concatenate([audio_features.flatten(), openl3_avg])
            return combined
        else:
            logger.warning("Failed to extract one or both embeddings")
            return None

    def analyze_raga_structure(self, audio, sr):
        """Analyze the structural elements of a raga performance"""
        embeddings = self.get_combined_embeddings(audio, sr)
        if embeddings is None:
            return None
            
        # Normalize embeddings
        embeddings = (embeddings - np.mean(embeddings)) / np.std(embeddings)
        
        # Extract additional musical features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        chromagram = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        return {
            'embeddings': embeddings,
            'embedding_dim': len(embeddings),
            'tempo': tempo,
            'tonal_features': np.mean(chromagram, axis=1),
            'model_types': ['mel_mfcc', 'openl3']
        }

def get_phrase_similarities(emb1, emb2):
    """Compare two phrases using their embeddings"""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))