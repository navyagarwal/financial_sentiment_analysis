import pandas as pd
from huggingface_hub import InferenceClient
from sklearn.metrics import classification_report
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Dataset/test_dataset.csv')

# Initialize the Hugging Face Inference Client
client = InferenceClient(api_key="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")

# Function to get sentiment prediction
def get_sentiment_prediction(instruction, input_text):
    prompt = f"{instruction} {input_text}"
    messages = [{"role": "user", "content": prompt}]
    
    try:
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        
        response_text = ""
        for chunk in stream:
            response_text += chunk.choices[0].delta.content
        
        # Check for sentiment in response
        for sentiment in ["positive", "neutral", "negative"]:
            if sentiment in response_text.lower():
                return sentiment
    except Exception as e:
        print(f"Error: {e}")
    
    return "neutral"  # Default to neutral if there's an issue

# Arrays for predictions and true labels
y_preds = []
y_true = df['output'].tolist()

# Iterate over the first row and get predictions
for _, row in df.iterrows():
    instruction = row['instruction']
    input_text = row['input']
    prediction = get_sentiment_prediction(instruction, input_text)
    y_preds.append(prediction)

# Print predictions
# print(y_preds)

# Save predictions to a file
with open("predictions_llama.txt", "w") as pred_file:
    for pred in y_preds:
        pred_file.write(pred + "\n")

# Generate and save classification report
report = classification_report(y_true, y_preds, labels=["negative", "neutral", "positive"], zero_division=0)

print("TEST ACCURACY")
print("=" * 22)
print(report)

with open("classification_report_llama.txt", "w") as report_file:
    report_file.write("TEST ACCURACY\n")
    report_file.write("=" * 22 + "\n")
    report_file.write(report)