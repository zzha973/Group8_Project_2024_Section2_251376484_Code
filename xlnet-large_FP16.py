import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    return correct_predictions.double() / total_predictions


def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    best_accuracy = 0.0
    accuracy_values = []
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_accuracy = compute_accuracy(model, val_loader, device)

        print(f'\nEpoch {epoch + 1} finished. Train Loss: {avg_train_loss:.4f}, Test Accuracy: {val_accuracy:.4f}')

        accuracy_values.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'xlnet-large.pt')
            print("Model saved!")
            print("\n")
        else:
            print("This round dropped")
            print("\n")

    return accuracy_values



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 25


    dataset = load_dataset('ag_news')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_labels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-6)


    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(tokenized_datasets['test'], batch_size=128,num_workers=8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)




    accuracy_values = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(range(1, epochs + 1), [x.cpu().numpy() for x in accuracy_values], 'b-o')

    plt.title('xlnet-large_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.grid(True)
    plt.show()
    plt.savefig('xlnet-large_accuracy.png', bbox_inches='tight')
    print(accuracy_values)


if __name__ == "__main__":
    main()
