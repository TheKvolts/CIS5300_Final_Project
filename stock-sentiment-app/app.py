import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    layout="wide"
)

# ============================================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================================


@st.cache_resource
def load_sentiment_model():
    """Load the fine-tuned FinBERT model from Hugging Face Hub for sentiment analysis"""
    sentiment_model_name = "suha-memon/finbert-stock-sentiment"

    try:
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        model.eval()  # Set to evaluation mode
        return tokenizer, model, sentiment_model_name
    except Exception as e:
        st.error(f"Error loading sentiment model from Hugging Face Hub: {e}")
        st.info(
            f"Make sure the model exists at: https://huggingface.co/{sentiment_model_name}")
        return None, None, None


sentiment_tokenizer, sentiment_model, sentiment_model_name = load_sentiment_model()


@st.cache_resource
def load_aspect_model():
    """
    Load the ABSA-FinBERT aspect classification model from Hugging Face Hub.

    This model classifies financial headlines into:
    Corporate, Economy, Market, and Stock.
    """
    aspect_model_name = "nick-cirillo/finbert-fiqa-aspect"

    try:
        tokenizer = AutoTokenizer.from_pretrained(aspect_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(aspect_model_name)
        model.eval()
        # Explicit aspect mapping (do not rely on base FinBERT config labels)
        id2label = {
            0: "Corporate",
            1: "Economy",
            2: "Market",
            3: "Stock"
        }
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}

        return tokenizer, model, id2label, aspect_model_name
    except Exception as e:
        st.error(f"Error loading aspect model from Hugging Face Hub: {e}")
        st.info(
            f"Make sure the model exists at: https://huggingface.co/{aspect_model_name}")
        return None, None, None, None

# ============================================================================
# ALPHA VANTAGE API FUNCTIONS
# ============================================================================


def get_news_for_ticker(ticker, api_key, limit=50):
    """
    Fetch news articles for a given stock ticker using Alpha Vantage API

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        api_key: Your Alpha Vantage API key
        limit: Maximum number of articles to return

    Returns:
        List of dictionaries containing article information
    """
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'apikey': api_key,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            st.error(f"API Error: {data['Error Message']}")
            return []

        if 'Note' in data:
            st.warning(f"API Limit: {data['Note']}")
            return []

        # Extract articles
        articles = []
        for article in data.get('feed', []):
            articles.append({
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'source': article.get('source', 'Unknown'),
                'time_published': article.get('time_published', ''),
                'summary': article.get('summary', '')
            })

        return articles

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

# ============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================================


def analyze_headline(text, tokenizer, model):
    """
    Analyze sentiment of a single headline

    Returns:
        Dictionary with probabilities and scores
    """
    if not text or not tokenizer or not model:
        return None

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Convert to dict
    result = {
        'negative': probs[0].item(),
        'neutral': probs[1].item(),
        'positive': probs[2].item(),
    }

    # Calculate derived metrics
    result['sentiment_score'] = result['positive'] - result['negative']
    result['confidence'] = max(
        result['negative'], result['neutral'], result['positive'])
    result['predicted_label'] = ['Negative',
                                 'Neutral', 'Positive'][probs.argmax().item()]

    return result


def calculate_overall_sentiment(all_results):
    """
    Calculate aggregate sentiment from multiple headlines
    Uses confidence-weighted averaging
    """
    if not all_results:
        return None

    total_score = 0
    total_weight = 0

    for result in all_results:
        confidence = result['confidence']
        score = result['sentiment_score']

        total_score += score * confidence
        total_weight += confidence

    weighted_avg = total_score / total_weight if total_weight > 0 else 0

    # Calculate average probabilities
    avg_negative = sum(r['negative'] for r in all_results) / len(all_results)
    avg_neutral = sum(r['neutral'] for r in all_results) / len(all_results)
    avg_positive = sum(r['positive'] for r in all_results) / len(all_results)

    return {
        'weighted_score': weighted_avg,
        'avg_negative': avg_negative,
        'avg_neutral': avg_neutral,
        'avg_positive': avg_positive,
        'num_headlines': len(all_results)
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_sentiment_gauge(score):
    """Create a gauge chart for overall sentiment"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Sentiment Score", 'font': {'size': 20}},
        delta={'reference': 0, 'increasing': {
            'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.3], 'color': '#ffcccc'},
                {'range': [-0.3, 0.3], 'color': '#ffffcc'},
                {'range': [0.3, 1], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'size': 14}
    )

    return fig


def create_probability_distribution(avg_negative, avg_neutral, avg_positive):
    """Create a bar chart of average probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Neutral', 'Positive'],
            y=[avg_negative, avg_neutral, avg_positive],
            marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f'],
            text=[f'{avg_negative:.1%}',
                  f'{avg_neutral:.1%}', f'{avg_positive:.1%}'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Average Sentiment Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# ============================================================================
# ASPECT CLASSIFICATION FUNCTIONS
# ============================================================================
def analyze_aspect(text, tokenizer, model, id2label):
    """
    Classify the aspect of a single financial headline.

    Returns:
        Dictionary with predicted aspect and confidence.
    """
    if not text or tokenizer is None or model is None or not id2label:
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probs).item())

    return {
        "aspect": id2label.get(predicted_idx, str(predicted_idx)),
        "confidence": probs[predicted_idx].item()
    }


def calculate_aspect_distribution(aspect_results):
    """
    Calculate counts and distribution for aspect classifications.
    """
    if not aspect_results:
        return None
    
    # Always track all four aspects, even if count is zero
    all_aspects = ["Corporate", "Economy", "Market", "Stock"]
    counts = {aspect: 0 for aspect in all_aspects}
    
    for result in aspect_results:
        aspect = result["aspect"]
        if aspect in counts:
            counts[aspect] += 1
        else:
            # In case a new/unknown label appears, add it
            counts[aspect] = counts.get(aspect, 0) + 1
    
    total = len(aspect_results)
    distribution = {aspect: count / total for aspect, count in counts.items()}

    return {
        "counts": counts,
        "distribution": distribution,
        "total": total
    }

# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("Stock Sentiment Analyzer")
    st.markdown(
        "### Analyze news sentiment and aspects for any stock using fine-tuned FinBERT models")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Get your free API key at https://www.alphavantage.co/support/#api-key"
        )

        if not api_key:
            st.info("Enter your API key to get started")
            st.markdown(
                "[Get a free API key](https://www.alphavantage.co/support/#api-key)")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool uses fine-tuned **FinBERT** models to analyze financial news:
        - **Sentiment** (Positive / Neutral / Negative)
        - **Aspect** (Corporate / Economy / Market / Stock)
        
        **App Authored By:**
        Suha Memon
        
        Utilizing Huggingface Models by:
        - Suha Memon
        - Nick Cirillo
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            placeholder="e.g., AAPL, TSLA, MSFT",
            help="Enter any valid stock ticker symbol"
        ).upper()

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        st.markdown(
            "Click **Analyze** to see both sentiment and aspect analysis results side by side.")

    # Single Analyze button
    analyze_button = st.button(
        "Analyze", type="primary", use_container_width=True)

    if analyze_button:
        # Validate models
        if sentiment_tokenizer is None or sentiment_model is None:
            st.error("Sentiment model failed to load. Please check the setup.")
            return

        aspect_tokenizer, aspect_model, aspect_id2label, aspect_model_name = load_aspect_model()
        if aspect_tokenizer is None or aspect_model is None:
            st.error("Aspect model failed to load. Please check the setup.")
            return

        if not api_key:
            st.error("Please enter your Alpha Vantage API key in the sidebar")
            return

        if not ticker:
            st.error("Please enter a stock ticker")
            return

        # Fetch news once
        with st.spinner(f"Fetching news for {ticker}..."):
            articles = get_news_for_ticker(
                ticker, api_key, limit=50)

        if not articles:
            st.warning(f"No news articles found for {ticker}")
            return

        st.success(f"Found {len(articles)} articles for {ticker}")

        # Analyze sentiment using sentiment model
        with st.spinner(f"Analyzing sentiment using {sentiment_model_name}..."):
            results = []
            for article in articles:
                sentiment = analyze_headline(
                    article['title'], sentiment_tokenizer, sentiment_model)
                if sentiment:
                    results.append({
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url'],
                        'time': article['time_published'],
                        **sentiment
                    })

        # Analyze aspects using aspect model
        with st.spinner(f"Classifying aspects using {aspect_model_name}..."):
            aspect_results = []
            for article in articles:
                aspect = analyze_aspect(
                    article['title'], aspect_tokenizer, aspect_model, aspect_id2label)
                if aspect:
                    aspect_results.append({
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url'],
                        'time': article['time_published'],
                        **aspect
                    })

        if not results or not aspect_results:
            st.error("Failed to analyze articles")
            return

        # Calculate overall sentiment
        overall = calculate_overall_sentiment(results)
        summary = calculate_aspect_distribution(aspect_results)

        # Calculate sentiment counts
        sentiment_counts = {
            "Positive": sum(1 for r in results if r['predicted_label'] == 'Positive'),
            "Neutral": sum(1 for r in results if r['predicted_label'] == 'Neutral'),
            "Negative": sum(1 for r in results if r['predicted_label'] == 'Negative')
        }
        sentiment_distribution = {
            label: count / len(results) if results else 0 
            for label, count in sentiment_counts.items()
        }

        # Define colors for each aspect (used in chart and table)
        aspect_colors = {
            "Corporate": "#4A90E2",  # Blue
            "Economy": "#50C878",    # Green
            "Market": "#FF6B35",     # Orange
            "Stock": "#9B59B6"       # Purple
        }
        
        # Define colors for sentiment
        sentiment_colors = {
            "Positive": "#6bcf7f",   # Green
            "Neutral": "#ffd93d",    # Yellow
            "Negative": "#ff6b6b"   # Red
        }

        st.markdown("---")
        st.header(f"Analysis Results for {ticker}")

        # ======================================================================
        # OVERALL ASSESSMENT AND GAUGE CHART
        # ======================================================================
        assessment_col1, gauge_col2 = st.columns([1, 2])
        
        with assessment_col1:
            # Overall Assessment
            st.markdown("#### Overall Assessment")
            
            if overall['weighted_score'] > 0.3:
                st.success(f"**Strong Positive Sentiment**")
                st.markdown(
                    f"Score: **{overall['weighted_score']:.3f}**. "
                    f"{overall['avg_positive']:.0%} positive headlines."
                )
            elif overall['weighted_score'] > 0.1:
                st.info(f"**Slightly Positive Sentiment**")
                st.markdown(f"Score: **{overall['weighted_score']:.3f}**")
            elif overall['weighted_score'] > -0.1:
                st.info(f"**Neutral Sentiment**")
                st.markdown(f"Score: **{overall['weighted_score']:.3f}**")
            elif overall['weighted_score'] > -0.3:
                st.warning(f"**Slightly Negative Sentiment**")
                st.markdown(f"Score: **{overall['weighted_score']:.3f}**")
            else:
                st.error(f"**Strong Negative Sentiment**")
                st.markdown(
                    f"Score: **{overall['weighted_score']:.3f}**. "
                    f"{overall['avg_negative']:.0%} negative headlines."
                )
        
        with gauge_col2:
            # Overall sentiment gauge chart
            gauge_fig = create_sentiment_gauge(overall['weighted_score'])
            st.plotly_chart(gauge_fig, use_container_width=True)

        # ======================================================================
        # SIDE BY SIDE RESULTS
        # ======================================================================
        
        # Main results columns - Sentiment on left, Aspect on right
        main_col1, main_col2 = st.columns(2)

        # ----------------------------------------------------------------------
        # LEFT COLUMN: SENTIMENT ANALYSIS
        # ----------------------------------------------------------------------
        with main_col1:
            st.subheader("Sentiment Analysis")

            # Overall metrics
            sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)

            with sent_col1:
                st.metric(
                    "Articles",
                    overall['num_headlines']
                )

            with sent_col2:
                st.metric(
                    "Score",
                    f"{overall['weighted_score']:.3f}",
                    help="Range: -1 (very negative) to +1 (very positive)"
                )

            with sent_col3:
                st.metric(
                    "Positive",
                    f"{overall['avg_positive']:.1%}"
                )

            with sent_col4:
                st.metric(
                    "Negative",
                    f"{overall['avg_negative']:.1%}"
                )

            # Visualizations
            dist_fig = create_probability_distribution(
                overall['avg_negative'],
                overall['avg_neutral'],
                overall['avg_positive']
            )
            st.plotly_chart(dist_fig, use_container_width=True)

            # Sentiment counts
            st.markdown("#### Sentiment Counts")
            for sentiment, count in sentiment_counts.items():
                color = sentiment_colors.get(sentiment, "#808080")
                st.markdown(
                    f"- <span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>{sentiment}</span>: {count} ({sentiment_distribution[sentiment]:.0%})",
                    unsafe_allow_html=True
                )

            # Download option
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Sentiment Results",
                data=csv,
                file_name=f"{ticker}_sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ----------------------------------------------------------------------
        # RIGHT COLUMN: ASPECT CLASSIFICATION
        # ----------------------------------------------------------------------
        with main_col2:
            st.subheader("Aspect Classification")

            # Overall metrics
            aspect_col1, aspect_col2, aspect_col3, aspect_col4 = st.columns(4)

            with aspect_col1:
                st.metric("Articles", summary['total'])

            with aspect_col2:
                st.write("")  # Spacing

            with aspect_col3:
                st.write("")  # Spacing

            with aspect_col4:
                st.write("")  # Spacing

            # Aspect distribution chart (matching sentiment chart style)
            aspects = list(summary['distribution'].keys())
            values = [summary['distribution'][a] for a in aspects]
            colors = [aspect_colors.get(a, "#808080") for a in aspects]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=aspects,
                    y=values,
                    marker_color=colors,
                    text=[f'{v:.1%}' for v in values],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Aspect Distribution",
                yaxis_title="Share of Headlines",
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Aspect counts with colored labels
            st.markdown("#### Aspect Counts")
            for aspect, count in summary['counts'].items():
                color = aspect_colors.get(aspect, "#808080")
                st.markdown(
                    f"- <span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>{aspect}</span>: {count} ({summary['distribution'][aspect]:.0%})",
                    unsafe_allow_html=True
                )

            # Download option
            df_aspects = pd.DataFrame(aspect_results)
            csv_aspects = df_aspects.to_csv(index=False)
            st.download_button(
                label="Download Aspect Results",
                data=csv_aspects,
                file_name=f"{ticker}_aspect_classification_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ======================================================================
        # INDIVIDUAL HEADLINES - FULL WIDTH BELOW
        # ======================================================================
        st.markdown("---")
        st.header("Individual Headlines")

        # Headlines side by side
        headlines_col1, headlines_col2 = st.columns(2)

        with headlines_col1:
            st.markdown("#### Sentiment Analysis")
            # Create DataFrame
            df = pd.DataFrame(results)

            # Format for display
            display_df = df[['title', 'predicted_label', 'negative',
                             'neutral', 'positive', 'confidence', 'source']].copy()
            display_df.columns = ['Headline', 'Sentiment', 'Negative',
                                  'Neutral', 'Positive', 'Confidence', 'Source']

            # Format percentages
            for col in ['Negative', 'Neutral', 'Positive', 'Confidence']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")

            # Color code sentiment
            def color_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #d4edda'
                elif val == 'Negative':
                    return 'background-color: #f8d7da'
                else:
                    return 'background-color: #fff3cd'

            styled_df = display_df.style.applymap(
                color_sentiment, subset=['Sentiment'])

            st.dataframe(styled_df, use_container_width=True, height=400)

        with headlines_col2:
            st.markdown("#### Aspect Classification")
            df_aspects = pd.DataFrame(aspect_results)
            display_df_aspects = df_aspects[[
                'title', 'aspect', 'confidence', 'source']].copy()
            display_df_aspects.columns = [
                'Headline', 'Aspect', 'Confidence', 'Source']
            display_df_aspects['Confidence'] = display_df_aspects['Confidence'].apply(
                lambda x: f"{x:.1%}")

            # Color code aspects
            def color_aspect(val):
                color = aspect_colors.get(val, "#FFFFFF")
                return f'background-color: {color}; color: white; font-weight: bold'

            styled_df_aspects = display_df_aspects.style.applymap(
                color_aspect, subset=['Aspect'])

            st.dataframe(styled_df_aspects,
                         use_container_width=True, height=400)

        # ======================================================================
        # RAW DATA SECTION
        # ======================================================================
        st.markdown("---")
        st.header("Raw Data (JSON)")
        st.markdown("Expand the sections below to see the raw JSON results (model input/output only)")
        
        # Prepare clean input/output pairs for sentiment
        sentiment_io = []
        for result in results:
            sentiment_io.append({
                "input": result['title'],  # Model input
                "output": {
                    "negative": result['negative'],
                    "neutral": result['neutral'],
                    "positive": result['positive'],
                    "predicted_label": result['predicted_label'],
                    "sentiment_score": result['sentiment_score'],
                    "confidence": result['confidence']
                }
            })
        
        # Prepare clean input/output pairs for aspect
        aspect_io = []
        for result in aspect_results:
            aspect_io.append({
                "input": result['title'],  # Model input
                "output": {
                    "aspect": result['aspect'],
                    "confidence": result['confidence']
                }
            })
        
        raw_col1, raw_col2 = st.columns(2)
        
        with raw_col1:
            with st.expander("Raw Sentiment Results (JSON)", expanded=False):
                st.json({
                    "model_name": sentiment_model_name,
                    "overall_sentiment": overall,
                    "individual_results": sentiment_io
                })
                
        with raw_col2:
            with st.expander("Raw Aspect Results (JSON)", expanded=False):
                st.json({
                    "model_name": aspect_model_name,
                    "aspect_distribution": summary,
                    "individual_results": aspect_io
                })
        
        # Raw articles from API
        with st.expander("Raw Articles from Alpha Vantage API (JSON)", expanded=False):
            st.json(articles)


if __name__ == "__main__":
    main()
