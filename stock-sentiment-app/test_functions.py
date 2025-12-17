#!/usr/bin/env python3
"""
Test script to test the sentiment and aspect analysis functions directly.
Run this to see raw results without the Streamlit UI.

Usage:
    python test_functions.py
"""

import json
from app import (
    load_model,
    load_aspect_model,
    get_news_for_ticker,
    analyze_headline,
    analyze_aspect,
    calculate_overall_sentiment,
    calculate_aspect_distribution
)

def test_sentiment_analysis():
    """Test sentiment analysis on a sample headline"""
    print("=" * 80)
    print("TESTING SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Load model
    print("\n[1] Loading sentiment model...")
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        print("❌ Failed to load sentiment model")
        return
    print("✅ Model loaded successfully")
    
    # Test on sample headlines
    test_headlines = [
        "Apple stock surges 10% after strong earnings report",
        "Tech stocks plummet as market crashes",
        "Microsoft announces quarterly results"
    ]
    
    print(f"\n[2] Testing on {len(test_headlines)} sample headlines...")
    results = []
    
    for headline in test_headlines:
        print(f"\n   Analyzing: '{headline}'")
        result = analyze_headline(headline, tokenizer, model)
        if result:
            results.append({
                'headline': headline,
                **result
            })
            print(f"   ✅ Result: {result['predicted_label']} "
                  f"(Positive: {result['positive']:.2%}, "
                  f"Neutral: {result['neutral']:.2%}, "
                  f"Negative: {result['negative']:.2%})")
    
    # Calculate overall
    if results:
        print("\n[3] Calculating overall sentiment...")
        overall = calculate_overall_sentiment(results)
        print(f"   ✅ Overall weighted score: {overall['weighted_score']:.3f}")
        print(f"   ✅ Average probabilities - Positive: {overall['avg_positive']:.2%}, "
              f"Neutral: {overall['avg_neutral']:.2%}, "
              f"Negative: {overall['avg_negative']:.2%}")
    
    # Print raw JSON
    print("\n[4] Raw JSON Results:")
    print(json.dumps({
        "overall": overall if results else None,
        "individual_results": results
    }, indent=2, default=str))
    
    return results, overall


def test_aspect_classification():
    """Test aspect classification on a sample headline"""
    print("\n" + "=" * 80)
    print("TESTING ASPECT CLASSIFICATION")
    print("=" * 80)
    
    # Load model
    print("\n[1] Loading aspect model...")
    tokenizer, model, id2label = load_aspect_model()
    if tokenizer is None or model is None:
        print("❌ Failed to load aspect model")
        return
    print("✅ Model loaded successfully")
    
    # Test on sample headlines
    test_headlines = [
        "Apple announces new iPhone launch",
        "Federal Reserve raises interest rates",
        "Stock market reaches all-time high",
        "Tesla shares jump 15% on delivery news"
    ]
    
    print(f"\n[2] Testing on {len(test_headlines)} sample headlines...")
    results = []
    
    for headline in test_headlines:
        print(f"\n   Analyzing: '{headline}'")
        result = analyze_aspect(headline, tokenizer, model, id2label)
        if result:
            results.append({
                'headline': headline,
                **result
            })
            print(f"   ✅ Aspect: {result['aspect']} "
                  f"(Confidence: {result['confidence']:.2%})")
    
    # Calculate distribution
    if results:
        print("\n[3] Calculating aspect distribution...")
        distribution = calculate_aspect_distribution(results)
        print(f"   ✅ Total articles: {distribution['total']}")
        print("   ✅ Aspect counts:")
        for aspect, count in distribution['counts'].items():
            print(f"      - {aspect}: {count} ({distribution['distribution'][aspect]:.1%})")
    
    # Print raw JSON
    print("\n[4] Raw JSON Results:")
    print(json.dumps({
        "distribution": distribution if results else None,
        "individual_results": results
    }, indent=2, default=str))
    
    return results, distribution


def test_alpha_vantage_api(api_key, ticker="AAPL", limit=5):
    """Test Alpha Vantage API (requires API key)"""
    import requests
    
    print("\n" + "=" * 80)
    print("TESTING ALPHA VANTAGE API")
    print("=" * 80)
    
    if not api_key:
        print("\n⚠️  No API key provided. Skipping API test.")
        print("   To test the API, set your API key:")
        print("   python test_functions.py YOUR_API_KEY")
        return None
    
    print(f"\n[1] Fetching news for {ticker} (limit: {limit})...")
    
    # Call API directly (bypassing Streamlit functions)
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
            print(f"❌ API Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            print(f"⚠️  API Limit: {data['Note']}")
            return None
        
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
        
        if not articles:
            print("❌ No articles returned")
            return None
        
        print(f"✅ Found {len(articles)} articles")
        
        # Print first article as example
        if articles:
            print("\n[2] Sample article (first one):")
            print(json.dumps(articles[0], indent=2, default=str))
        
        # Print all articles
        print("\n[3] All articles (raw JSON):")
        print(json.dumps(articles, indent=2, default=str))
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def main():
    """Main test function"""
    import sys
    
    # Get API key from command line if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("\n" + "=" * 80)
    print("STOCK SENTIMENT ANALYZER - FUNCTION TESTING")
    print("=" * 80)
    
    # Test sentiment analysis
    sentiment_results, sentiment_overall = test_sentiment_analysis()
    
    # Test aspect classification
    aspect_results, aspect_distribution = test_aspect_classification()
    
    # Test API (if key provided)
    articles = test_alpha_vantage_api(api_key)
    
    # Full integration test (if API key provided)
    if api_key and articles:
        print("\n" + "=" * 80)
        print("FULL INTEGRATION TEST")
        print("=" * 80)
        
        print("\n[1] Running full analysis on fetched articles...")
        tokenizer, model = load_model()
        aspect_tokenizer, aspect_model, aspect_id2label = load_aspect_model()
        
        if tokenizer and model and aspect_tokenizer and aspect_model:
            # Analyze first article
            if articles:
                article = articles[0]
                print(f"\n   Article: '{article['title']}'")
                
                # Sentiment
                sentiment = analyze_headline(article['title'], tokenizer, model)
                if sentiment:
                    print(f"   ✅ Sentiment: {sentiment['predicted_label']} "
                          f"(Score: {sentiment['sentiment_score']:.3f})")
                
                # Aspect
                aspect = analyze_aspect(article['title'], aspect_tokenizer, aspect_model, aspect_id2label)
                if aspect:
                    print(f"   ✅ Aspect: {aspect['aspect']} "
                          f"(Confidence: {aspect['confidence']:.2%})")
                
                # Full result
                print("\n[2] Full raw result for this article:")
                print(json.dumps({
                    "article": article,
                    "sentiment": sentiment,
                    "aspect": aspect
                }, indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\nTo see raw results in the Streamlit UI, run:")
    print("  streamlit run app.py")
    print("\nThen click 'Analyze' and expand the 'Raw Data (JSON)' section.")


if __name__ == "__main__":
    main()

