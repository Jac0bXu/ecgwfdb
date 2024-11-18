import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt  # For confusion matrix plotting

# Custom Dataset class to handle CSV file loading
class CSVDataset(Dataset):
    def __init__(self, file_paths, labels):
        """
        Initializes the dataset with file paths and their labels.
        Args:
            file_paths (list): List of CSV file paths.
            labels (list): List of labels corresponding to each file.
        """
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of files in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads a single sample (CSV file) based on its index.
        Args:
            idx (int): Index of the file to load.
        Returns:
            data_tensor (Tensor): The numerical data from the CSV file.
            label_tensor (Tensor): The corresponding label as a tensor.
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        # Load only the first 3 columns from the CSV file
        data = np.loadtxt(file_path, delimiter=',', usecols=(0, 1, 2))
        # Convert data and label to PyTorch tensors
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor

# Function to prepare data loaders for training and validation
# Updated prepare_data function
def prepare_data(data_dir):
    """
    Prepares the data loaders for training and validation datasets.
    Args:
        data_dir (str): Directory containing the CSV files.
    Returns:
        train_loader, val_loader: Data loaders for training and validation datasets.
    """
    file_paths = []  # List to store paths of all CSV files
    labels = []  # List to store labels

    # Updated label extraction logic
    def extract_label(file_name):
        base_name = os.path.basename(file_name).lower()
        if base_name.startswith("none"):
            return 0  # Label for NONE
        elif base_name.startswith("other"):
            return 1  # Label for OTHER
        else:
            return -1  # Unknown label

    # Iterate through all files in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):  # Process only CSV files
            label = extract_label(file_name)  # Extract label using updated logic
            if label != -1:  # Ignore files with unknown labels
                file_paths.append(os.path.join(data_dir, file_name))
                labels.append(label)

    print(f'labels: {labels}')
    print(f'# of labels: {len(labels)}')

    # Split data into training (80%) and validation (20%) sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = CSVDataset(train_files, train_labels)
    val_dataset = CSVDataset(val_files, val_labels)
    print(f'train dataset: {train_labels}')
    # Create data loaders for efficient batch processing
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader


# Define the RNN-based classification model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the RNN model.
        Args:
            input_size (int): Number of input features per time step (e.g., 3 columns).
            hidden_size (int): Number of hidden units in the RNN.
            num_layers (int): Number of stacked RNN layers.
            output_size (int): Number of output classes (e.g., 2 for binary classification).
        """
        super(RNNClassifier, self).__init__()
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Define the fully connected layer to produce output logits
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            out (Tensor): Output logits of shape (batch_size, output_size).
        """
        # Pass input through RNN; h_n is the last hidden state
        _, h_n = self.rnn(x)  # h_n shape: (num_layers, batch_size, hidden_size)
        # Use the last hidden state's last layer for classification
        out = self.fc(h_n[-1])
        return out

# Train the model and validate it
def train_model(train_loader, val_loader, input_size, hidden_size, num_layers, epochs=10, learning_rate=0.01):
    """
    Trains and validates the RNN model.
    Args:
        train_loader, val_loader: Data loaders for training and validation data.
        input_size (int): Number of features per time step.
        hidden_size (int): Number of hidden units in the RNN.
        num_layers (int): Number of RNN layers.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        model: The trained RNN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = RNNClassifier(input_size, hidden_size, num_layers, output_size=2).to(device)  # Initialize model
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Store all predictions and true labels for confusion matrix
    all_labels = []
    all_predictions = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0  # Initialize loss for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # Create progress bar

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item()  # Accumulate loss
            progress_bar.set_postfix(loss=train_loss / len(progress_bar))  # Update progress bar with loss

        # Validation step
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get class predictions
                total += labels.size(0)  # Total samples
                correct += (predicted == labels).sum().item()  # Count correct predictions
                all_labels.extend(labels.cpu().numpy())  # Store true labels
                all_predictions.extend(predicted.cpu().numpy())  # Store predicted labels

        # Print validation accuracy for the epoch
        print(f"Validation Accuracy: {correct / total:.4f}")

    # Display the confusion matrix
    if all_labels and all_predictions:  # Ensure there are labels and predictions
        cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])  # Compute confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['none', 'other'])  # Format matrix
        disp.plot(cmap="Blues")  # Plot the confusion matrix with a blue colormap
        disp.ax_.set_title("Confusion Matrix")  # Set the plot title
        disp.ax_.set_xlabel("Predicted Label")  # Label for x-axis
        disp.ax_.set_ylabel("True Label")  # Label for y-axis
        plt.show()  # Force the plot to display
    else:
        print("No predictions or labels to display in the confusion matrix.")

    return model

# Main function
if __name__ == "__main__":
    # Specify the directory containing CSV files
    data_directory = "C:/Users/jxxzh/PycharmProjects/ecgwfdb/newecgdata/data1"  # Update with your actual directory path

    # Prepare data loaders for training and validation
    train_loader, val_loader = prepare_data(data_directory)

    # Train the RNN model
    input_size = 3  # Number of input features per time step (columns in the CSV)
    hidden_size = 50  # Number of hidden units in RNN
    num_layers = 2   # Number of stacked RNN layers
    trained_model = train_model(train_loader, val_loader, input_size, hidden_size, num_layers,epochs=2)
