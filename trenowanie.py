import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def read_data(file_name):
    reviews = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            review = json.loads(line)
            reviews.append(review['text'])
            labels.append(review['label'])
    return reviews, labels

def train_model(model, train_loader, optimizer, device, num_epochs=3):
    for epoch in range(num_epochs):
        train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}')
    return model

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    total_accuracy = 0
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_accuracy += (outputs[1].argmax(1) == labels).cpu().numpy().mean()
        
    return total_accuracy / len(data_loader)

def eval_model(model, data_loader, device):
    model = model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_accuracy += (outputs[1].argmax(1) == labels).cpu().numpy().mean()

    return total_accuracy / len(data_loader)

def predict_sentiment(model, tokenizer, text, max_len=128):
    model = model.eval()

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

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    # Load data
    reviews, labels = read_data('simplified_reviews.json')
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.1)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128


    train_dataset = ReviewDataset(train_reviews, train_labels, tokenizer, max_len)
    test_dataset = ReviewDataset(test_reviews, test_labels, tokenizer, max_len)


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model_path = "bert_sentiment_model.pth"

    if os.path.exists(model_path):
        model = load_model(model, model_path)
        print("Model loaded from disk.")
    else:
        model = train_model(model, train_loader, optimizer, device)
        save_model(model, model_path)
        print("Model trained and saved to disk.")

    test_acc = eval_model(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')

    while True:
        new_review = input("Enter a review (or type 'exit' to quit): ")
        if new_review.lower() == 'exit':
            break
        predicted_label = predict_sentiment(model, tokenizer, new_review)
        if predicted_label == 1:
            label = "pozytywna"
        else:
            label = "negatywna"

        print(f'Predicted label for the review: {label}')
