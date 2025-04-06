import numpy as np
from utils.logger import setup_logger
from scipy.stats import mode
import pickle

logger = setup_logger("RagaRecognition")

def extract_pitch_features(pitch_data):
    """Extract relevant features from pitch data"""
    frequencies = np.array(pitch_data['frequency'])
    confidence = np.array(pitch_data['confidence'])
    
    # Filter out low confidence predictions
    mask = confidence > 0.8
    clean_frequencies = frequencies[mask]
    
    if len(clean_frequencies) == 0:
        return None
    
    # Convert frequencies to midi notes
    midi_notes = 69 + 12 * np.log2(clean_frequencies / 440)
    midi_notes = np.round(midi_notes)
    
    # Get the pitch distribution
    pitch_hist = np.zeros(128)  # Full MIDI range
    for note in midi_notes:
        if 0 <= note < 128:  # Valid MIDI range
            pitch_hist[int(note)] += 1
            
    # Normalize
    pitch_hist = pitch_hist / np.sum(pitch_hist)
    
    return {
        'pitch_histogram': pitch_hist,
        'main_notes': mode(midi_notes)[0]
    }

def match_raga(features, raga_templates):
    """Match pitch features against raga templates"""
    if features is None:
        return []
    
    scores = {}
    for raga_name, template in raga_templates.items():
        # Compare pitch histograms
        hist_diff = np.sum((features['pitch_histogram'] - template['pitch_histogram'])**2)
        # Check if main notes are present in raga's characteristic phrases
        note_match = len(set(features['main_notes']).intersection(template['characteristic_notes']))
        
        # Combined score (lower is better)
        scores[raga_name] = hist_diff - (0.5 * note_match)
    
    return sorted(scores.items(), key=lambda x: x[1])

def recognize_raga(pitch_data, raga_templates):
    """Main function for raga recognition"""
    try:
        features = extract_pitch_features(pitch_data)
        if features is None:
            logger.warning("Could not extract reliable pitch features")
            return None
            
        results = match_raga(features, raga_templates)
        
        logger.info("Raga Recognition Results:")
        for raga, score in results[:3]:
            logger.info(f"{raga}: {score:.3f}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in raga recognition: {e}")
        return None
