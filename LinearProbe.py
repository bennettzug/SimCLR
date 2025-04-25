import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Import your model architecture
from models.resnet_simclr import ResNetSimCLR


class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def extract_features(loader, model, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features"):
            images = images.to(device)

            # Extract features (without the projection head)
            # Get the features directly from the layer before FC
            x = images
            # Forward pass through all layers except the final fc
            for name, module in model.backbone.named_children():
                if name != "fc":  # Skip the projection head
                    x = module(x)
                else:
                    break

            feature_vectors = x.view(x.size(0), -1)  # Flatten features

            features.append(feature_vectors.cpu())
            labels.append(targets)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    print(f"Feature shape after extraction: {feature_vectors.shape}")
    return features, labels


def evaluate(probe_model, features, labels, device):
    probe_model.eval()
    features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = probe_model(features)
        _, predicted = torch.max(outputs.data, 1)

    accuracy = (predicted == labels).sum().item() / labels.size(0)

    # Move tensors back to CPU for sklearn metrics
    y_true = labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    return accuracy, classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation for SimCLR")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--data", type=str, default="./datasets", help="path to dataset")
    parser.add_argument("--dataset-name", default="stl10", choices=["stl10", "cifar10"], help="dataset name")
    parser.add_argument("--batch-size", default=256, type=int, help="batch size")
    parser.add_argument("--workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs for linear probe training")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--arch", default="resnet18", help="model architecture")
    parser.add_argument("--out-dim", default=128, type=int, help="feature dimension")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    print(f"Using device: {device}")

    # Load the SimCLR model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # Freeze the encoder
    for param in model.parameters():
        param.requires_grad = False

    # Prepare datasets
    # Use transform without augmentations for evaluation
    if args.dataset_name == "cifar10":
        img_size = 32
        num_classes = 10

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

        train_dataset = datasets.CIFAR10(root=args.data, train=True, transform=transform, download=True)

        test_dataset = datasets.CIFAR10(root=args.data, train=False, transform=transform, download=True)

    elif args.dataset_name == "stl10":
        img_size = 96
        num_classes = 10

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

        train_dataset = datasets.STL10(root=args.data, split="train", transform=transform, download=True)

        test_dataset = datasets.STL10(root=args.data, split="test", transform=transform, download=True)

    # Split train set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("Extracting features from the training set...")
    train_features, train_labels = extract_features(train_loader, model, device)

    # Determine feature dimension from the extracted features
    feature_dim = train_features.shape[1]
    print(f"Detected feature dimension: {feature_dim}")

    print("Extracting features from the validation set...")
    val_features, val_labels = extract_features(val_loader, model, device)

    print("Extracting features from the test set...")
    test_features, test_labels = extract_features(test_loader, model, device)

    print(f"Train features shape: {train_features.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    print(f"Unique train labels: {torch.unique(train_labels)}")
    print(f"Unique validation labels: {torch.unique(val_labels)}")
    print(f"Unique test labels: {torch.unique(test_labels)}")
    print(f"Number of classes in linear probe: {num_classes}")
    # Create and train the linear probe model
    probe_model = LinearProbe(feature_dim, num_classes).to(device)
    print(f"Linear probe input dimension: {probe_model.classifier.in_features}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    best_val_acc = 0.0

    print(f"Starting linear probe training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        probe_model.train()
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()
        outputs = probe_model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # Validation
        val_acc, _, _ = evaluate(probe_model, val_features, val_labels, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(probe_model.state_dict(), "best_linear_probe.pth")

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}"
        )

    # Load best model for evaluation
    probe_model.load_state_dict(torch.load("best_linear_probe.pth"))

    # Final evaluation on test set
    test_acc, report, conf_matrix = evaluate(probe_model, test_features, test_labels, device)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    if args.dataset_name == "cifar10":
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif args.dataset_name == "stl10":
        class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = conf_matrix[i, i]
        class_total = conf_matrix[i, :].sum()
        print(f"{class_name}: {class_correct / class_total:.4f}")


if __name__ == "__main__":
    main()
