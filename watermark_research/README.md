
# Watermark Research

### Setup
```
conda create -n audio-watermark python=3.12
conda activate audio-watermark
pip install -r requirements.txt
```

### Testing
```
cd watermark_research/src
python watermark_testing.py
```

#### Command Arguments Description

- `--mode`: Choose between `"benchmark"` (default) to run the full watermark survivability test or `"detector"` to only check watermark detection without attack.
- `--audio_dir`: Path to the directory containing input audio files (e.g., `../../dataset/LibriSpeech`).
- `--output_dir`: Directory where benchmark results and artifacts will be saved (default: `../results`).
- `--watermarks`: List of watermarking methods to test. Available options:
  - `AudioSeal`
  - `WavMark`
  - `SilentCipher`
  - `SemanticPCA`
  - `SemanticCluster`
  - `SemanticRandom`
- `--filecount`: Number of audio files to process from the dataset (default: 10).

Example usage:
```
# Run benchmark mode
python watermark_testing.py --mode benchmark --audio_dir ../../dataset/LibriSpeech --output_dir ../results --watermarks SemanticCluster SemanticRandom --filecount 10

# Run detector checker mode
python watermark_testing.py --mode detector --audio_dir ../../dataset/LibriSpeech --watermarks SemanticCluster SemanticRandom --filecount 10
```

The initial run of the experiment is stored in `./results/LibriSpeech-init`.