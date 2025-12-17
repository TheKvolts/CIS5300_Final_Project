# Stock Sentiment Analyzer

An interactive Streamlit app that uses fine-tuned FinBERT models to analyze news sentiment and aspect classification for any stock ticker. It fetches recent news from the Alpha Vantage API and provides:
- **Sentiment Analysis**: Classifies headlines as Positive, Neutral, or Negative
- **Aspect Classification**: Categorizes headlines into Corporate, Economy, Market, or Stock aspects

## Requirements

- Python 3.9+ (recommended)
- `pip`
- An Alpha Vantage API key (free): https://www.alphavantage.co/support/#api-key
- Internet access (to download Hugging Face models and call the Alpha Vantage API)

## Setup

From the project root:

```bash
cd stock-sentiment-app
```

### Create and activate a virtual environment

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install `streamlit`, `transformers`, `torch`, `pandas`, `plotly`, `requests`, and other required packages.

## Running the app

From the `stock-sentiment-app` directory with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a URL in the terminal, typically:

```text
Local URL: http://localhost:8501
```

Open that URL in your browser to use the app.

## Using the app

1. In the sidebar, paste your Alpha Vantage API key.
2. Choose how many news articles to analyze (5-50).
3. Enter a stock ticker (e.g., `AAPL`, `TSLA`, `MSFT`) and click **Analyze**.
4. View:
   - **Left side**: Overall sentiment score and distribution (gauge chart + probability bar chart)
   - **Right side**: Aspect classification results (distribution chart + counts)
   - **Below**: Per-headline sentiment and aspect tables
   - **Download options**: CSV files for both sentiment and aspect results
   - **Raw JSON data**: Expand the "Raw Data (JSON)" section at the bottom to see model inputs/outputs

## Testing Functions & Viewing Raw Results

### Option 1: View Raw Results in the UI

After clicking **Analyze** in the Streamlit app, scroll down to the **"Raw Data (JSON)"** section. Expand the sections to see:
- Raw sentiment results (overall + individual headline results)
- Raw aspect classification results (distribution + individual results)
- Raw articles from the Alpha Vantage API

### Option 2: Test Functions Directly (Command Line)

Run the test script to test functions and see raw JSON output without the UI:

```bash
# Test without API (uses sample headlines)
python test_functions.py

# Test with API (requires your Alpha Vantage API key)
python test_functions.py YOUR_API_KEY
```

The test script will:
- Test sentiment analysis on sample headlines
- Test aspect classification on sample headlines
- Test the Alpha Vantage API (if API key provided)
- Print all results as formatted JSON
- Run a full integration test combining all functions

**Note**: The test script may need to be updated if function names in `app.py` have changed. The main functions are:
- `load_sentiment_model()` - Loads the sentiment model
- `load_aspect_model()` - Loads the aspect model
- `analyze_headline()` - Analyzes sentiment of a headline
- `analyze_aspect()` - Classifies aspect of a headline
- `get_news_for_ticker()` - Fetches news from Alpha Vantage API

## Model Details

The app automatically downloads two fine-tuned FinBERT models from Hugging Face:

1. **Sentiment Model**: `suha-memon/finbert-stock-sentiment`
   - Classifies sentiment as: Negative (0), Neutral (1), Positive (2)
   - Performance: 81.88% accuracy, F1-Macro: 0.8009

2. **Aspect Model**: `nick-cirillo/finbert-fiqa-aspect`
   - Classifies aspects as: Corporate (0), Economy (1), Market (2), Stock (3)
   - Trained on FiQA-2018 dataset for financial aspect classification

You do **not** need to manually place model files in the repo; they are downloaded automatically on first run and cached by Streamlit.

## Project Structure

```
stock-sentiment-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── test_functions.py     # Test script for command-line testing
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── venv/                # Virtual environment (not in git)
```

## Troubleshooting

### Model Loading Issues

If models fail to load:
- Check your internet connection
- Verify the model names exist on Hugging Face:
  - https://huggingface.co/suha-memon/finbert-stock-sentiment
  - https://huggingface.co/nick-cirillo/finbert-fiqa-aspect
- Models are cached after first download, so subsequent runs should be faster

### API Key Issues

- Make sure your Alpha Vantage API key is valid
- Free tier has rate limits (5 API calls per minute, 500 per day)
- If you see rate limit errors, wait a minute and try again

### Dependencies Issues

If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```

## License

Part of the CIS5300 NLP Final Project.

