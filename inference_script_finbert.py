import pandas as pd
import requests
from sklearn.metrics import classification_report
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {"Authorization": "Bearer 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"}

# Function to query the FinBERT model
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to get sentiment prediction
def get_sentiment_prediction(input_text):
    try:
        result = query({"inputs": input_text})
        
        # Extract sentiment
        if 'error' not in result:
            predictions = result[0]
            # Sort predictions by confidence score and take the label with the highest score
            predicted_label = sorted(predictions, key=lambda x: x['score'], reverse=True)[0]['label']
            return predicted_label.lower()
        else:
            print(f"Error in response: {result['error']}")
    except Exception as e:
        print(f"Error: {e}")
    
    return "neutral"  # Default to neutral if there's an issue

# Load the dataset
df = pd.read_csv('Dataset/test_dataset.csv')

# Arrays for predictions and true labels
y_preds = []
y_true = df['output'].tolist()

# Iterate over the first 3 rows and get predictions
for _, row in df.iterrows():
    input_text = row['input']
    prediction = get_sentiment_prediction(input_text)
    y_preds.append(prediction)

# Print predictions
# print(y_preds)

# Save predictions to a file
with open("predictions_finbert.txt", "w") as pred_file:
    for pred in y_preds:
        pred_file.write(pred + "\n")

# Generate and save classification report
report = classification_report(y_true, y_preds, labels=["negative", "neutral", "positive"], zero_division=0)

print("TEST ACCURACY")
print("=" * 22)
print(report)

with open("classification_report_finbert.txt", "w") as report_file:
    report_file.write("TEST ACCURACY\n")
    report_file.write("=" * 22 + "\n")
    report_file.write(report)