import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define a function to compute the accuracy of predictions
def compute_accuracy(model, data_loader, device):
    model.eval()  # Set model to evaluation mode to disable dropout
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No gradient computation for efficiency
        for batch in data_loader:
            # Load input tensors to the specified device (GPU/CPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Perform a forward pass and get the predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            # Calculate the number of correct predictions
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    # Compute accuracy as a ratio of correct predictions to total predictions
    return correct_predictions.double() / total_predictions

# Function to handle training and evaluation of the model
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs=4):
    best_accuracy = 0.0
    accuracy_values = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            optimizer.zero_grad()  # Reset gradients accumulation

            # Load batch data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Compute loss and perform a backward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate schedule

        avg_train_loss = total_loss / len(train_loader)
        val_accuracy = compute_accuracy(model, val_loader, device)

        # Logging training and validation results
        print(f'\nEpoch {epoch + 1} finished. Train Loss: {avg_train_loss:.4f}, Test Accuracy: {val_accuracy:.4f}')

        accuracy_values.append(val_accuracy)

        # Save model if the accuracy is the best observed
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'bert-base_accuracy(FP32).pt')
            print("Model saved!")
        else:
            print("This round dropped")

    return accuracy_values

# Main function to initialize and run the training process
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Setup computation device

    dataset = load_dataset('ag_news')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)

    # Tokenization function to preprocess text data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create DataLoader for handling batches
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(tokenized_datasets['test'], batch_size=128, num_workers=8)

    optimizer = AdamW(model.parameters(), lr=2e-6)  # Initialize the optimizer
    epochs = 25
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model and plot the accuracy results
    accuracy_values = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(range(1, epochs + 1), [x.cpu().numpy() for x in accuracy_values], 'b-o')
    plt.title('bert-base_accuracy(FP32)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    plt.savefig('bert-base_accuracy(FP32).png', bbox_inches='tight')
    print(accuracy_values)

if __name__ == "__main__":
    main()
