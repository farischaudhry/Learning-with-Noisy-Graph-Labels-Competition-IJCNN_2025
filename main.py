import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from loadData import GraphDataset
import os
import pandas as pd
import wandb    
import glob

def add_zeros(data):
    """Add zero features to nodes (necessary for embedding layers)."""
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data

def evaluate(model, data_loader, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            if calculate_accuracy and data.y is not None:
                correct += (pred == data.y).sum().item()
                total += data.y.numel()

    if calculate_accuracy:
        accuracy = correct / total if total > 0 else 0
        return accuracy, predictions
    return predictions

def save_model(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")

def get_latest_checkpoint(checkpoint_dir, dataset_name):
    """Finds the latest (highest epoch) checkpoint for a dataset."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"model_{dataset_name}_epoch_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found for {dataset_name}. Train the model first.")
    
    # Extract epoch numbers & get the highest one
    checkpoint_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]), reverse=True)
    return checkpoint_files[0]

def train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs, log_path, checkpoint_dir, dataset_name, patience=10):
    
    best_val_acc = 0.0
    patience_counter = 0
    
    with open(log_path, "a") as log_file:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()   

                pred = output.argmax(dim=1)
                correct_train += (pred == data.y).sum().item()
                total_train += data.y.numel()

            train_loss = total_loss / len(train_loader)
            train_acc = correct_train / total_train if total_train > 0 else 0
            val_acc, _ = evaluate(model, val_loader, device, calculate_accuracy=True)

            # Log accuracy and loss. Every 10 epochs print and log locally. Save model checkpoint every 10 epochs.
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
                log_file.write(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}\n")

                checkpoint_path = os.path.join(checkpoint_dir, f"model_{dataset_name}_epoch_{epoch+1}.pth")
                save_model(model, checkpoint_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                log_file.write(f"Early stopping at epoch {epoch + 1}\n")
                break

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    input_dim = 300
    hidden_dim = 64
    num_epochs = 1000
    learning_rate = 0.001
    batch_size = 128
    output_dim = 6  # Fixed 

    model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Train nwe model on training set
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        
        # Split into train & validation (80% train, 20% val)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Initialize WandB (only if training)
        wandb.init(project="graph-noisy-labels", group=os.getenv("WANDB_RUN_GROUP", "unknown"))
        wandb.config.update({"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "epochs": num_epochs, "lr": learning_rate})

        os.makedirs("logs", exist_ok=True)
        dataset_name = os.path.basename(os.path.dirname(args.train_path))
        log_path = f"logs/training_log_{dataset_name}.txt"
        checkpoint_dir = "checkpoints"

        # Train model
        train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs, log_path, checkpoint_dir, dataset_name)

        # Save final trained model
        dataset_name = os.path.basename(os.path.dirname(args.train_path))
        checkpoint_path = f"checkpoints/model_{dataset_name}_epoch_{num_epochs}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        save_model(model, checkpoint_path)

        # Finish WandB run
        wandb.finish()
    # Load pre-trained model for inference
    else:
        dataset_name = os.path.basename(os.path.dirname(args.test_path))
        checkpoint_path = get_latest_checkpoint("checkpoints", dataset_name)
        load_model(model, checkpoint_path, device)

    # Evaluate test set 
    predictions = evaluate(model, test_loader, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))

    # Save predictions to CSV
    output_dir = "submission"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"testset_{dataset_name}.csv")
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    # Random seeds for reproducibility
    SEED = 1729
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)
