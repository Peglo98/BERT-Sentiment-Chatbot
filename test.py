import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import json

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

def eval_model(model, data_loader, device):
    model = model.eval()
    total_accuracy = 0
    num_batches = len(data_loader)
    with torch.no_grad():
        for i, d in enumerate(tqdm(data_loader, desc="Evaluating")):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_accuracy += (outputs[1].argmax(1) == labels).cpu().numpy().mean()
            
            current_accuracy = total_accuracy / (i + 1)
            print(f"Current accuracy after batch {i + 1}/{num_batches}: {current_accuracy * 100:.2f}%")
            
    return total_accuracy / len(data_loader)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def test_on_new_dataset(model_path, test_data_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)
    model = load_model(model, model_path)
    print("Model loaded from disk.")

    reviews, labels = read_data(test_data_path)
    test_dataset = ReviewDataset(reviews, labels, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=16)

    test_acc = eval_model(model, test_loader, device)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

if __name__ == "__main__":
    model_path = "bert_sentiment_model.pth"
    test_data_path = "output.json"

    test_on_new_dataset(model_path, test_data_path)
