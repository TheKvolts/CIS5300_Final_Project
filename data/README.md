# NLP 5300 Final Project Dataset

## Financial Sentiment Analysis Dataset — Overview

**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis  

A 3-class sentiment classification dataset consisting of short financial news headlines.

---

## 📝 Dataset Summary
| Property | Value |
|----------|-------|
| Total samples | **5,322** headlines |
| Task | Sentiment classification |
| Labels | `negative`, `neutral`, `positive` (mapped to 0, 1, 2) |
| File format | CSV |


| Column | Type | Description |
|--------|------|-------------|
| `Sentence` | string | Raw news headline text |
| `Sentiment` | string | Class label (`negative`, `neutral`, `positive`) |
| `Label`| int | encoding for `Sentiment` |


## Example

**Sentence**  
> "The GeoSolutions technology will leverage Benefon’s GPS solutions by providing Location Based Search ..."

**Sentiment**  
> `positive`

**Label**
> `2`

---

## Data Directory Structure

The dataset (5322 examples) is split into **train**, **development**, and **test** sets using an 80/10/10 ratio.

| Split | Count | File |
|--------|-------|------|
| Train | **4,673** | `data/train/train.csv` |
| Development (Val) | **584** | `data/development/development.csv` |
| Test | **585** | `data/test/test.csv` |

All splits preserve the original column structure and label mapping.

## Notes
- Labels have been encoded for model usage:  
  - `negative → 0`
  - `neutral → 1`
  - `positive → 2`
- seed = 42
- No further preprocessing (tokenization, cleaning, etc.) has been applied yet
