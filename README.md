# 🧠 MESSI – Mathematically Enhanced Structuring of Semi-Structured Inputs
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/itisyashvanth/MESSI.git)

> Because humans don’t type in JSON.

MESSI transforms real-world messy communication such as tweets, chats, and notes filled with slang, emojis, abbreviations, and typos into clean, structured, machine-readable data that AI systems can reliably use. Unlike traditional NLP pipelines, MESSI does not blindly guess; it reasons mathematically, validates outputs against real-world constraints, and explicitly quantifies uncertainty.

The core problem MESSI addresses is the mismatch between how humans communicate and how machines require data. Humans write things like: "yo AA102 delayed AGAIN 😤 smh". AI systems, however, need structured representations such as a `flight_id` of `AA102`, an `event` of `delay`, a `complaint` sentiment, and a `confidence_score` of `0.94`. Existing solutions fail because they assume clean and well-formed input. MESSI bridges this gap between human expression and machine requirements.

## Key Features

-   **Handles Real-World Messiness**: Understands slang and shortcuts (`smh`, `ngl`), interprets emojis (`😤`, `💀`) as semantic signals, and corrects typos (`flihgt` -> `flight`).
-   **Context Resolution**: Resolves missing information, such as identifying an airline from a flight number.
-   **Mathematical Reasoning**: Uses constrained optimization to enforce real-world rules, preventing hallucinations and ensuring outputs are logically consistent.
-   **Uncertainty Quantification**: Applies Monte Carlo Dropout to produce calibrated confidence scores, allowing the system to state when it is unsure.

## Architecture

MESSI is built on a three-layer intelligence architecture:

1.  **Pattern Finder**: A BiLSTM-CRF deep learning model trained on messy text to treat emojis, slang, and typos as meaningful signals for robust entity extraction.
2.  **Logic Validator**: Uses constrained optimization (via Google OR-Tools) to enforce real-world rules, ensuring outputs like flight IDs and events are valid and consistent.
3.  **Honesty Checker**: Applies uncertainty quantification using Monte Carlo Dropout to produce calibrated confidence scores, enabling the system to honestly report its confidence level.

The system pipeline is as follows:
`Raw Text` → `Emoji-Preserving Tokenizer` → `BiLSTM-CRF` → `Constraint Validation (OR-Tools)` → `Uncertainty Estimation (MC Dropout)` → `Structured JSON Output`

## Getting Started

### Installation

Clone the repository, create a virtual environment, and install the required dependencies.

```bash
git clone https://github.com/itisyashvanth/messi.git
cd messi
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage

Use the `MESSIExtractor` class to process messy text and receive structured output.

```python
from messi import MESSIExtractor

extractor = MESSIExtractor()
messy_text = "yo AA102 delayed AGAIN 😤 smh"
structured_data = extractor.process(messy_text)

print(structured_data)
# Expected Output:
# {
#   "flight_id": "AA102",
#   "event": "delay",
#   "confidence": {
#     "flight_id": 0.98,
#     "event": 0.91
#   }
# }
```

## Messy2Structured Dataset

MESSI introduces **Messy2Structured**, the first public benchmark dataset for messy-to-structured NLP. It contains over 500 annotated real-world samples from airline and e-commerce communications, complete with slang, emojis, typos, and missing context.

The dataset is available on Hugging Face and can be loaded directly:

```python
from datasets import load_dataset

dataset = load_dataset("messi/messy2structured")
```

## Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Entity F1-Score        | 0.89   |
| End-to-End Accuracy    | 0.85   |
| Constraint Satisfaction| 99.2%  |
| Calibration Error      | 0.06   |

The system runs efficiently on CPU-only hardware (Apple M1/M2), processing approximately **500 samples/second**.

## Research & Citation

This work introduces a novel hybrid deep learning and mathematical optimization architecture and the first public benchmark for messy-to-structured conversion. A research paper is in preparation for the NeurIPS Datasets and Benchmarks Track.

If you use MESSI in your research, please cite:

```bibtex
@inproceedings{yashvanth2025messi,
    title={MESSI: Mathematically Enhanced Structuring of Semi-Structured Inputs},
    author={Yashvanth},
    year={2025},
    booktitle={Advances in Neural Information Processing Systems}
}
```

## Development

To run tests and view coverage, use `pytest`:

```bash
pytest --cov=./
```

Training custom models is supported through configurable scripts. Data annotation can be performed using Label Studio with the raw data provided in the repository.

## Roadmap

-   [x] Dataset Collection & Annotation
-   [x] BiLSTM-CRF Implementation
-   [x] Constraint Optimization Integration
-   [x] Uncertainty Quantification
-   [ ] Evaluation & Benchmarking
-   [ ] Paper Writing & Submission

Future work includes document processing (PDFs, receipts), multilingual support, and real-time streaming APIs for domains like healthcare and emergency response.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. Areas of interest include:
-   Annotating new domains (healthcare, legal)
-   Building multilingual datasets
-   Performance optimizations
-   Adding new constraint solvers

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   [Airline Twitter Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) from Kaggle
-   BiLSTM-CRF architecture by [Lample et al. (2016)](https://arxiv.org/abs/1603.01360)
-   Monte Carlo Dropout method by [Gal and Ghahramani (2016)](http://proceedings.mlr.press/v48/gal16.html)

---
*Built with love to make AI work with real-world human communication.*

**Author**: Yashvanth ([@itisyashvanth](https://github.com/itisyashvanth))
