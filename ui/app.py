import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_input import preprocess
from models.neural_analysis import AudioEmbedder
from symbolic_parser.stage2_raga_recognition import recognize_raga
from utils.create_raga_templates import create_raga_templates, save_templates
import numpy as np
import librosa
import pickle

class RagaAnalysisUI:
    def __init__(self):
        self.audio_embedder = AudioEmbedder()
        self.ensure_raga_templates()
        self.load_raga_templates()

    def ensure_raga_templates(self):
        """Create raga templates if they don't exist"""
        template_path = "data/raga_templates.pkl"
        if not os.path.exists(template_path):
            print("No raga templates found. Creating templates...")
            templates = create_raga_templates()
            save_templates(templates)

    def load_raga_templates(self):
        try:
            with open("data/raga_templates.pkl", 'rb') as f:
                self.raga_templates = pickle.load(f)
                print("Successfully loaded raga templates")
        except Exception as e:
            print(f"Error loading raga templates: {e}")
            self.raga_templates = None

    def analyze_audio(self, audio_path):
        if not audio_path:
            return "Please upload an audio file"
            
        try:
            # Status updates
            print("Starting audio analysis...")
            print(f"Processing file: {audio_path}")
            
            # Load and preprocess audio
            segments = preprocess.preprocess_audio(audio_path)
            if not segments:
                return "Error: Failed to process audio file. Please ensure it's a valid audio recording."

            print(f"Found {len(segments)} stable segments")
            results = []
            
            for idx, segment in enumerate(segments):
                print(f"Analyzing segment {idx + 1}/{len(segments)}")
                
                try:
                    # Get neural embeddings
                    y, sr = librosa.load(audio_path, sr=None)
                    start_sample = int(segment['start_time'] * sr)
                    end_sample = int(segment['end_time'] * sr)
                    segment_audio = y[start_sample:end_sample]
                    
                    # Neural analysis
                    neural_analysis = self.audio_embedder.analyze_raga_structure(segment_audio, sr)
                    
                    # Traditional raga recognition
                    if self.raga_templates:
                        raga_results = recognize_raga(segment['pitch_data'], self.raga_templates)
                        if raga_results:
                            top_raga, score = raga_results[0]
                            # Get the top 3 raga matches
                            top_3_ragas = "\n".join([f"  {raga}: {1-s:.2f} confidence" 
                                                   for raga, s in raga_results[:3]])
                        else:
                            top_raga, score = "Unknown", 0.0
                            top_3_ragas = "No clear matches found"
                    else:
                        top_raga, score = "No templates loaded", 0.0
                        top_3_ragas = "Raga templates not available"

                    # Format detailed results
                    results.append(
                        f"Segment {idx + 1}:\n"
                        f"Time Range: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s\n"
                        f"Duration: {segment['end_time'] - segment['start_time']:.2f}s\n"
                        f"Tempo: {segment['tempo']:.1f} BPM\n"
                        f"Top Raga Matches:\n{top_3_ragas}\n"
                        f"Musical Features:\n"
                        f"- Tonal Center: {segment['tonal_features']['chroma'].argmax()}\n"
                        f"- Pitch Stability: {np.mean(segment['pitch_data']['confidence']):.2f}\n"
                        "-------------------"
                    )
                except Exception as segment_error:
                    results.append(f"Error analyzing segment {idx + 1}: {str(segment_error)}")
                    continue

            print("Analysis complete")
            return "\n".join(results)

        except Exception as e:
            error_msg = str(e)
            print(f"Error during analysis: {error_msg}")
            return f"Error during analysis: {error_msg}\nPlease ensure the audio file is a valid Indian classical music recording."

    def create_interface(self):
        iface = gr.Interface(
            fn=self.analyze_audio,
            inputs=[
                gr.Audio(
                    type="filepath", 
                    label="Upload Indian Classical Music"
                )
            ],
            outputs=[
                gr.Textbox(
                    label="Analysis Results", 
                    lines=15,
                    placeholder="Analysis results will appear here..."
                )
            ],
            title="Indian Classical Music Analyzer",
            description="""Upload an Indian classical music recording for detailed analysis.
            The system will detect stable segments, analyze the raga structure, and extract musical features.
            Supported formats: WAV, MP3, OGG, etc.""",
            examples=[
                ["sample_audio/classical_music.wav"],
                ["sample_audio/classical-instrumental-emotional-music-300927.mp3"]
            ],
            allow_flagging="never"
        )
        return iface

def launch_ui():
    app = RagaAnalysisUI()
    interface = app.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    launch_ui()