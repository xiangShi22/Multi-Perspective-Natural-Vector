import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os
from sklearn.metrics import classification_report, accuracy_score

# Define neural network (remove logic related to factor_size)
class WeightedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, min_weight_value=1e-5):
        super(WeightedNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.weights = nn.Parameter(torch.randn(input_dim))  
        self.min_weight_value = min_weight_value

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        weighted_input = x * self.weights  
        out = self.fc1(weighted_input)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def constraint_weights(self):
        with torch.no_grad():
            self.weights.data = torch.clamp(self.weights.data, min=self.min_weight_value)


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder path
folder_path = "/covid_ncbi/2mer"

# Get all CSV file paths
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize empty lists for storing data and labels
data_list = []
label_list = []

# Read each CSV file
for idx, file in enumerate(csv_files):
    df = pd.read_csv(file)
    # Add label column, using the file index as label
    df['label'] = idx
    data_list.append(df)

# Concatenate all data
data = pd.concat(data_list, axis=0)

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values   # The last column is the label

# Undersampling
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=10)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

print(f'Number of samples after resampling: {X_resampled.shape[0]}')

# Standardize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Convert to PyTorch tensors and move to GPU
X_resampled = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_resampled = torch.tensor(y_resampled, dtype=torch.long).to(device)

input_dim = X_resampled.shape[1]
hidden_dim = 512 
output_dim = len(np.unique(y_resampled.cpu()))  

model = WeightedNeuralNetwork(input_dim, hidden_dim, output_dim).to(device)

# Loss function and optimizer
class_counts = torch.bincount(y_resampled)
class_weights = 1.0 / class_counts
loss_weights = class_weights.float().to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weights)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

total_accuracy = 0  

for fold, (train_index, val_index) in enumerate(kf.split(X_resampled)):
    print(f"\nFold {fold+1}/{5}")
    
    # Split training and validation sets
    X_train, X_val = X_resampled[train_index], X_resampled[val_index]
    y_train, y_val = y_resampled[train_index], y_resampled[val_index]
    
    # Train the model
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Force weights to remain non-zero
        model.constraint_weights()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
    
    # Evaluate model on validation set
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_val.cpu(), predicted.cpu())
        total_accuracy += accuracy  # Accumulate accuracy
    
    print(f"Validation Accuracy for Fold {fold+1}: {accuracy:.4f}")
    print(f"Classification Report for Fold {fold+1}:\n{classification_report(y_val.cpu(), predicted.cpu())}")
    
# Compute and print average accuracy
average_accuracy = total_accuracy / 5
print(f"\nAverage Accuracy: {average_accuracy:.4f}")
