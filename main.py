from audio_input import preprocess
from symbolic_parser.stage2_raga_recognition import recognize_raga
from models.neural_analysis import AudioEmbedder
from ui.app import launch_ui
import pickle
import logging
import argparse
import sys
import os

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def load_raga_templates(template_path):
    try:
        logger.debug(f"Loading templates from {template_path}")
        with open(template_path, 'rb') as f:
            templates = pickle.load(f)
            logger.debug(f"Loaded {len(templates)} raga templates")
            return templates
    except Exception as e:
        logger.error(f"Error loading raga templates: {e}")
        return None

def analyze_audio(audio_path, template_path="data/raga_templates.pkl"):
    """Main analysis function combining traditional and neural approaches"""
    logger.debug(f"Processing audio file: {audio_path}")
    
    # Initialize neural analysis
    audio_embedder = AudioEmbedder()
    
    # Preprocess audio
    segments = preprocess.preprocess_audio(audio_path)
    if not segments:
        logger.error("Audio preprocessing failed")
        return None

    print("Stage 1 (Preprocessing) completed successfully. Proceeding to analysis...")
    logger.debug(f"Found {len(segments)} segments")
    
    # Load raga templates
    raga_templates = load_raga_templates(template_path)
    if not raga_templates:
        logger.error("Failed to load raga templates")
        return None

    results = []
    for segment in segments:
        logger.debug(f"Analyzing segment {segment['segment_index']}")
        
        # Traditional raga recognition
        raga_result = recognize_raga(segment['pitch_data'], raga_templates)
        
        # Neural analysis
        neural_result = audio_embedder.analyze_raga_structure(
            segment['pitch_data']['frequency'],
            16000  # Default sample rate for analysis
        )
        
        if raga_result:
            top_raga, score = raga_result[0]
            results.append({
                'segment_index': segment['segment_index'],
                'time_range': (segment['start_time'], segment['end_time']),
                'raga': top_raga,
                'confidence': 1 - score,
                'neural_embedding_dim': neural_result['embedding_dim'] if neural_result else None,
                'tonal_features': segment['tonal_features']
            })
            
            print(f"Segment {segment['segment_index']}: "
                  f"Most likely raga: {top_raga} (confidence: {1-score:.3f})")
            logger.debug(f"Match details - Raga: {top_raga}, Score: {score}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Indian Classical Music Analysis System')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--ui', action='store_true', help='Launch web interface')
    args = parser.parse_args()

    if args.ui:
        launch_ui()
    elif args.audio:
        results = analyze_audio(args.audio)
        if not results:
            print("Analysis failed. Check logs for details.")
    else:
        print("Please provide an audio file (--audio) or launch the UI (--ui)")
        sys.exit(1)

if __name__ == "__main__":
    main()