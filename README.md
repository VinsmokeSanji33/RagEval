```markdown
# RagEval

## Overview

This project is designed to evaluate various metrics for different types of Retrieval-Augmented Generation (RAG) implementations. It helps identify the best RAG technique for a given dataset by assessing performance across multiple metrics.

## Project Structure


├── data
│   └── Place your dataset here.
├── src
    ├── evaluation_metrics
    │   └── Contains scripts for defining and computing evaluation metrics.
    ├── QsnGen
    │   └── Contains modules for generating question-answer pairs.
    ├── rag_evaluation
    │   └── Scripts for implementing and evaluating RAG techniques.
    ├── tools
        └── Utility scripts for preprocessing and other auxiliary tasks.
```

## Features

- **Custom Dataset Support**: Add your dataset in the `data` folder.
- **Question-Answer Pair Generation**: Automatically generates Q&A pairs for the dataset.
- **RAG Evaluation**: Supports multiple RAG techniques and metrics for comparison.
- **Flexible Metrics**: Evaluate using pre-defined or custom metrics.

## supported metrics
- **Correctness**: evaluate the relevance and correctness of a generated answer against a reference answer.
- **guidelines**: to evaluate a question answer given user specified guidelines.
- **Faithfullness**: to measure if the response from a query engine matches any source nodes.This is useful for measuring if the response was
hallucinated.
- **relevancy**: to measure if the response + source nodes match the query.
- **Note**: to add data specifc guidelines change the guidelines.txt in tools , place a every new guideline in a newline
## supported Rag
- Currently this support naive rag evulation 

## Prerequisites

- Python
- OpenAI API Key (required in the `.env` file)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VinsmokeSanji33/RagEval.git
   cd RagEval
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up the `.env` file:
   - Add your OpenAI API Key:
     ```plaintext
     OPEN_AI_KEY=your_openai_api_key
     ```

## Usage

1. Place your dataset in the `data` folder.
2. Run the main script to start the evaluation:
   ```bash
   python src/main.py
   ```

## Output

- The evaluation results will be displayed in the console.

## Contribution

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest features.

## License

This project is licensed under the [MIT License](LICENSE).