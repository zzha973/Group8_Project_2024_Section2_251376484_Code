import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Define a function to compute the accuracy of the model
def compute_accuracy(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients for validation
        for batch in data_loader:
            # Move batch of data to the device (GPU or CPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass to get outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)  # Get the predictions

            # Calculate correct predictions
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    # Return accuracy as the ratio of correct to total predictions
    return correct_predictions.double() / total_predictions

# Define a function for training and evaluating the model
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    best_accuracy = 0.0
    accuracy_values = []

    scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

    for epoch in range(epochs):  # Loop over the dataset multiple times
        model.train()  # Set model to training mode
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            optimizer.zero_grad()  # Zero the parameter gradients

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():  # Mixed precision context
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()  # Scale loss and call backward to create scaled gradients
            scaler.step(optimizer)  # Scale down gradients and step optimizer
            scaler.update()  # Update the scale for next iteration
            scheduler.step()  # Adjust the learning rate based on the number of epochs

            total_loss += loss.item()  # Aggregate the loss

        avg_train_loss = total_loss / len(train_loader)
        val_accuracy = compute_accuracy(model, val_loader, device)

        print(f'\nEpoch {epoch + 1} finished. Train Loss: {avg_train_loss:.4f}, Test Accuracy: {val_accuracy:.4f}')

        accuracy_values.append(val_accuracy)  # Store accuracy for plotting

        if val_accuracy > best_accuracy:  # Save the best model
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'bert-base_accuracy(FP16).pt')
            print("Model saved!")
        else:
            print("This round dropped")

    return accuracy_values

# Define the main function to setup and run training and evaluation
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Setup device
    epochs = 25  # Number of training epochs

    # Load and preprocess the dataset
    dataset = load_dataset('ag_news')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-6)  # Setup optimizer

    # Function to tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create data loaders for training and testing
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(tokenized_datasets['test'], batch_size=128, num_workers=8)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train and evaluate the model
    accuracy_values = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs)

    # Plot and save the accuracy over epochs
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(range(1, epochs + 1), [x.cpu().numpy() for x in accuracy_values], 'b-o')
    plt.title('bert-base_accuracy(FP16)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    plt.savefig('bert-base_accuracy(FP16).png', bbox_inches='tight')
    print(accuracy_values)  # Print the final accuracy values

if __name__ == "__main__":
    main()
