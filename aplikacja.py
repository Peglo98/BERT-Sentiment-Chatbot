import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

def load_model(path):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model("bert_sentiment_model.pth")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    prediction = predict_sentiment(review)
    sentiment = 'pozytywna' if prediction == 1 else 'negatywna'
    return jsonify(sentiment=sentiment)

def predict_sentiment(text, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

if __name__ == '__main__':
    app.run(debug=True)
