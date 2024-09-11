# Sentiment Analysis using Huggingface

This repository contains a sentiment analysis model built using Huggingface's `AutoModelForSequenceClassification` and fine-tuned for text classification. The model can classify text sentiments using the `Igniter909/distilbert-base-uncased-finetuned-emotion` pre-trained model.

## Features
- **Fine-tuned model**: Uses the Huggingface library for fine-tuning on a specific dataset.
- **Custom sentiment analysis**: Includes functions for analyzing and predicting sentiments of custom text inputs.

## Model Details
The model is based on the `AutoModelForSequenceClassification` architecture, which is ideal for tasks involving text classification. It has been fine-tuned using Huggingface's pre-trained models, specifically the `transformersbook/distilbert-base-uncased-finetuned-emotion`.

### Dependencies
- Python 3.x
- Huggingface Transformers (`pip install transformers`)
- Pandas (`pip install pandas`)
- Matplotlib (`pip install matplotlib`)

## How to Run the Model

### Step 1: Install dependencies
```bash
pip install transformers pandas matplotlib
```

### Step 2: Load and use the model
The model can be used with the Huggingface pipeline for text classification. Example code to load the model:
```python
from transformers import pipeline

# Load the pre-trained model
model_id = 'Igniter909/distilbert-base-uncased-finetuned-emotion'
classifier = pipeline('text-classification', model=model_id)
```

### Step 3: Analyze a custom text
You can analyze the sentiment of custom text using the classifier:
```python
# Example text
custom_tweet = 'I saw a movie today and it was really good'
preds = classifier(custom_tweet, return_all_scores=True)
```

### Step 4: Visualize the sentiment
The code includes visualization of sentiment scores:
```python
import pandas as pd
import matplotlib.pyplot as plt

preds_df = pd.DataFrame(preds[0])
plt.bar(preds_df['label'], 100 * preds_df['score'], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
```

## Custom Function for Sentiment Analysis
A custom function `find_sentiment()` allows you to input any text and returns the most likely sentiment:
```python
def find_sentiment(Generate):
    if(Generate):
        text = input("Enter the text or tweet: ")
        preds = classifier(text, return_all_scores=True)
        data = pd.DataFrame(preds[0])
        max_label = data.loc[data['score'].idxmax(), 'label']
        print("The sentiment is:", max_label)

Generate = True
find_sentiment(Generate)
```

## License
This project is licensed under the MIT License.
