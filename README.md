# P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark

[![](https://img.shields.io/badge/arXiv-2505.17104-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2505.17104)

[![Dataset - P2PInstruct](https://img.shields.io/badge/Dataset-P2PInstruct-blue)](https://huggingface.co/datasets/ASC8384/P2PInstruct)
[![Dataset - P2PEval](https://img.shields.io/badge/Dataset-P2PEval-blue)](https://huggingface.co/datasets/ASC8384/P2PEval)

## 🚀 Try it on Hugging Face Spaces

This application is deployed on Hugging Face Spaces! You can try it directly in your browser without any installation:

**🎓 [Launch P2P Paper-to-Poster Generator](https://huggingface.co/spaces/ASC8384/P2P)**

### Quick Start on Spaces:
1. Upload your PDF research paper
2. Enter your OpenAI API key and base URL (if using proxy)
3. Input the AI model name (e.g., gpt-4o-mini, claude-3-sonnet)
4. Configure the figure detection service URL
5. Click "Generate Poster" and wait for processing
6. Preview the generated poster and download JSON/HTML files
7. Recommended to use Claude model for better performance

## Overview

P2P is an AI-powered tool that automatically converts academic research papers into professional conference posters. This repository contains the code for generating and evaluating these posters, leveraging large language models to extract key information and create visually appealing presentations.

The full research paper is available on [arXiv](https://arxiv.org/abs/2505.17104).

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
playwright install
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
@misc{sun2025p2pautomatedpapertopostergeneration,
      title={P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark}, 
      author={Tao Sun and Enhao Pan and Zhengkai Yang and Kaixin Sui and Jiajun Shi and Xianfu Cheng and Tongliang Li and Wenhao Huang and Ge Zhang and Jian Yang and Zhoujun Li},
      year={2025},
      eprint={2505.17104},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17104}, 
}
```
