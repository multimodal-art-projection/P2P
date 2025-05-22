# P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark

[![Dataset - P2PInstruct](https://img.shields.io/badge/Dataset-P2PInstruct-blue)](https://huggingface.co/datasets/ASC8384/P2PInstruct)
[![Dataset - P2PEval](https://img.shields.io/badge/Dataset-P2PEval-blue)](https://huggingface.co/datasets/ASC8384/P2PEval)


## Overview

P2P is an AI-powered tool that automatically converts academic research papers into professional conference posters. This repository contains the code for generating and evaluating these posters, leveraging large language models to extract key information and create visually appealing presentations.

The full research paper is available on [arXiv](https://arxiv.org/abs/XXXX.XXXXX).

**Note:** Due to the large size of the evaluation and training datasets, only simple samples are included in this repository. The complete datasets are available on HuggingFace:
- [P2PInstruct](https://huggingface.co/datasets/ASC8384/P2PInstruct) - Training dataset
- [P2PEval](https://huggingface.co/datasets/ASC8384/P2PEval) - Benchmark dataset

## Repository Structure

### Core Files
- `main.py`: Main entry point for generating a poster from a single paper
- `start.py`: Batch processing script for generating posters from multiple papers 
- `end.py`: Evaluation coordinator that processes generated posters
- `evalv2.py`: Core evaluation logic with metrics and comparison methods
- `figure_detection.py`: Utility for detecting and extracting figures from PDFs

### Directories
- `poster/`: Core poster generation logic
  - `poster.py`: Main poster generation implementation
  - `figures.py`: Figure extraction and processing utilities
  - `compress.py`: Image compression utilities
  - `loader.py`: PDF loading utilities

- `eval/`: Evaluation tools and resources
  - `eval_checklist.py`: Checklist-based evaluation implementation
  - `predict_with_xgboost.py`: ML-based poster quality prediction
  - `common.yaml`: Common evaluation parameters
  - `xgboost_model.joblib`: Pre-trained evaluation model

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating a Single Poster

To generate a poster from a single paper:

```bash
# Deploy figure_detection first
python main.py --url="URL_TO_PDF" --pdf="path/to/paper.pdf" --model="gpt-4o-mini" --output="output/poster.json"
```

#### Parameters:
- `--url`: URL for PDF processing service (detecting and extracting figures)
- `--pdf`: Path to the local PDF file
- `--model`: LLM model to use (default: gpt-4o-mini)
- `--output`: Output file path (default: poster.json)

#### Output Files:
- `poster.json`: JSON representation of the poster
- `poster.html`: HTML version of the poster
- `poster.png`: PNG image of the poster

### Batch Generating Posters

To generate posters for multiple papers:

1. Organize your papers in a directory structure:
```
eval/data/
  └─ paper_id_1/
     └─ paper.pdf
  └─ paper_id_2/
     └─ paper.pdf
  ...
```

2. Edit `start.py` to configure:
   - `url`: URL for PDF processing service
   - `input_dir`: Directory containing papers (default: "eval/data")
   - `models`: List of AI models to use for generation

3. Run the batch generation script:
```bash
python start.py
```

Generated posters will be saved to:
```
eval/temp-v2/{model_name}/{paper_id}/
  └─ poster.json
  └─ poster.html
  └─ poster.png
```

### Evaluating Posters

To evaluate generated posters:

1. Ensure reference materials exist:
```
eval/data/{paper_id}/
  └─ poster.png (reference poster)
  └─ checklist.yaml (evaluation checklist)
```

2. Run the evaluation script:
```bash
python end.py
```

Evaluation results will be saved to `eval/temp-v2/results.jsonl`.

## Citation

If you find our work useful, please consider citing P2P:

```bibtex

```
