import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AdamW
from tqdm import tqdm
import time
import os
from sklearn.utils.class_weight import compute_class_weight

start = time.time()

# Load the dataset
df = pd.read_csv("Book1.csv")

# Check unique labels in the dataset
unique_labels = df['label'].unique()
num_labels = len(unique_labels)
print("Number of unique labels:", num_labels)
print("Unique labels:", unique_labels)

# Verify label range
expected_labels = set(range(num_labels))
if set(unique_labels) != expected_labels:
    print("Error: Labels are not within the expected range.")
    # You may need to fix the labels in your dataset.

# Verify label encoding
if df['label'].isnull().sum() > 0:
    print("Error: There are missing values in the label column.")
    # You may need to handle missing values in your dataset.

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = self.data.iloc[idx]['label']

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

# Define training parameters
batch_size = 16
epochs = 10
learning_rate = 2e-5

# Create DataLoader for training and validation sets
train_dataset = CustomDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_df, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=unique_labels, y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the criterion with class weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Variables to keep track of the best model
best_accuracy = 0
best_cm = None
best_epoch = -1
save_directory = "./saved_model3"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Lists to store losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_train_preds = 0
    total_train_examples = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batches'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct_train_preds += (preds == labels).sum().item()
        total_train_examples += labels.size(0)

        # Diagnostic print statements
        if batch['input_ids'].shape[0] < 5:  # Only print for the first few small batches
            print(f'Batch logits: {outputs.logits}')
            print(f'Batch loss: {loss.item()}')

        loss.backward()
        optimizer.step()

    average_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train_preds / total_train_examples

    train_losses.append(average_train_loss)
    train_accuracies.append(train_accuracy)

    print(f'Training Loss: {average_train_loss}, Training Accuracy: {train_accuracy * 100}%')

    # Validation loop
    model.eval()
    total_val_loss = 0
    correct_val_preds = 0
    total_val_examples = 0

    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation', unit='batches'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_predictions.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            correct_val_preds += (preds == labels).sum().item()
            total_val_examples += labels.size(0)

    average_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val_preds / total_val_examples

    val_losses.append(average_val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Validation Loss: {average_val_loss}, Validation Accuracy: {val_accuracy * 100}%')

    # Generate confusion matrix
    cm = confusion_matrix(val_targets, val_predictions)

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_cm = cm
        best_epoch = epoch + 1
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("Best model saved with accuracy:", val_accuracy * 100)

print(f'Best epoch: {best_epoch} with accuracy: {best_accuracy * 100}%')

# Display the confusion matrix of the best epoch
plt.figure(figsize=(10, 7))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Best Epoch {best_epoch}')
plt.show()

# Plotting training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Plotting training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.legend()
plt.show()

print("Training completed.")
end = time.time()
print("Time:", (end-start), "seconds")