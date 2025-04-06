import numpy as np
import pickle
import os

def create_raga_templates():
    """Create templates for common Carnatic ragas"""
    
    # Define raga templates with their characteristic notes (in MIDI numbers)
    # Sa = 60 (Middle C), Re = 62, Ga = 64, Ma = 65, Pa = 67, Dha = 69, Ni = 71
    raga_templates = {
        'Mayamalavagowla': {
            'characteristic_notes': [60, 62, 63, 67, 69, 70],  # S R1 G3 P D1 N2
            'pitch_histogram': np.zeros(128)
        },
        'Shankarabharanam': {
            'characteristic_notes': [60, 62, 64, 65, 67, 69, 71],  # S R2 G3 M1 P D2 N3
            'pitch_histogram': np.zeros(128)
        },
        'Kalyani': {
            'characteristic_notes': [60, 62, 64, 66, 67, 69, 71],  # S R2 G3 M2 P D2 N3
            'pitch_histogram': np.zeros(128)
        },
        'Kharaharapriya': {
            'characteristic_notes': [60, 62, 63, 65, 67, 69, 70],  # S R2 G2 M1 P D2 N2
            'pitch_histogram': np.zeros(128)
        },
        'Thodi': {
            'characteristic_notes': [60, 61, 63, 65, 67, 68, 70],  # S R1 G2 M1 P D1 N2
            'pitch_histogram': np.zeros(128)
        }
    }

    # Create pitch histograms based on characteristic notes
    for raga_name, template in raga_templates.items():
        hist = np.zeros(128)
        notes = template['characteristic_notes']
        
        # Add primary notes with higher weight
        for note in notes:
            hist[note] = 1.0
            
        # Add octave relationships
        for note in notes:
            # Lower octave
            if note - 12 >= 0:
                hist[note - 12] = 0.5
            # Upper octave
            if note + 12 < 128:
                hist[note + 12] = 0.5
                
        # Normalize histogram
        hist = hist / np.sum(hist)
        template['pitch_histogram'] = hist

    return raga_templates

def save_templates(templates, output_dir="data"):
    """Save raga templates to a pickle file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raga_templates.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(templates, f)
    
    print(f"Saved raga templates to {output_path}")

if __name__ == "__main__":
    templates = create_raga_templates()
    save_templates(templates)