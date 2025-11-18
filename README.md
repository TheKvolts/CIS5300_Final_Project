# CIS5300_Final_Project

## Setup

To set up the project environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Milestone 2
A 1-2 page pdf report describing your evaluation measure, baselines, and their performance. Include any equations to you need to consisely, formally, describe all of these components.
You should describe your evaluation metric in a markdown file called scoring.md. Your scoring.md file should also show how to run your evaluation script on the command line (with example arguments, and example output).
You should include your evaluation script (you can call then score.py if you’re writing it in python).
You should upload simple-baseline.py and describe how to use the code in simple-baseline.md.
You should upload any code supporting your strong baseline and describe how to use it in strong-baseline.md.

## Metrics used: 
* Accuracy: Overall correctness
* F1-Score (Macro): Treats all classes equally - important because financial sentiment classes may be imbalanced
* F1-Score (Weighted): Accounts for class imbalance
* Per-Class Precision/Recall: Detailed breakdown for negative, neutral, positive
These metrics are standard for multi-class classification and were used in our literature review papers (Saleiro et al. 2017, Moore & Rayson 2017). F1-macro is particularly important because minority sentiment classes (likely "negative" in financial news) need fair evaluation.


## Plan for upcoming milestones
For M2, we will stick with single-label classification to establish a solid baseline. ABSA will be implemented as extensions in M3/M4 after discussing data requirements with our mentor.

