# Indian Classical Music Analysis System

A sophisticated system for analyzing Indian classical music recordings, with capabilities for raga recognition, tonal analysis, and structural segmentation.

## Features

- **Audio Preprocessing**
  - Silence removal and audio normalization
  - Advanced pitch tracking optimized for Indian classical music
  - Stable segment detection for identifying sustained notes
  - High-resolution spectral analysis

- **Raga Recognition**
  - Built-in templates for common Carnatic ragas
  - Pitch histogram analysis
  - Note pattern matching
  - Confidence scoring for raga identification

- **Neural Analysis**
  - Deep learning-based audio embeddings using OpenL3
  - Mel-spectrogram and MFCC feature extraction
  - Combined traditional and neural approaches for robust analysis

- **Musical Feature Extraction**
  - Tempo detection
  - Tonal center identification
  - Pitch stability analysis
  - Chromagram generation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/indian-music-analysis.git
   cd indian-music-analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/MacOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Analyze a single audio file:
```bash
python main.py --audio path/to/your/music.wav
```

### Web Interface

Launch the web interface for interactive analysis:
```bash
python main.py --ui
```

The web interface provides:
- Audio file upload
- Visualization of audio waveform and spectrogram
- Detailed analysis results including:
  - Detected raga with confidence scores
  - Tempo and rhythmic analysis
  - Tonal feature analysis
  - Segmentation results

## Project Structure

- `audio_input/` - Audio preprocessing and feature extraction
- `models/` - Neural network models and embeddings
- `symbolic_parser/` - Raga recognition and pattern matching
- `ui/` - Web interface implementation
- `utils/` - Helper functions and utilities
- `data/` - Raga templates and model data
- `sample_audio/` - Example audio files

## Supported Ragas

Currently includes templates for the following ragas:
- Mayamalavagowla (S R1 G3 P D1 N2)
- Shankarabharanam (S R2 G3 M1 P D2 N3)
- Kalyani (S R2 G3 M2 P D2 N3)
- Kharaharapriya (S R2 G2 M1 P D2 N2)
- Thodi (S R1 G2 M1 P D1 N2)

## Technical Details

- **Audio Processing**:
  - Sample Rate: Adaptive (preserves original)
  - Frequency Range: 50Hz - 1500Hz
  - Window Size: 2048 samples
  - Hop Length: 512 samples

- **Neural Features**:
  - OpenL3 embeddings (512-dimensional)
  - Mel-spectrogram (128 mel bands)
  - MFCC features (40 coefficients)

## Requirements

- Python 3.8+
- librosa>=0.9.2
- torch>=2.0.0
- openl3>=0.4.2
- scipy>=1.7.3
- numpy>=1.22.4
- gradio>=3.35.2

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [librosa](https://librosa.org/) for audio processing
- Uses [OpenL3](https://github.com/marl/openl3) for audio embeddings
- Web interface powered by [Gradio](https://gradio.app/)