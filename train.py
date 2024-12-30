import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\nEpoch {epoch + 1} Training:")
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
   
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    
    print(f"\nEpoch {epoch+1} Statistics:")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def evaluate_model(model, validation_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_predicts = []
    all_labels = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicts = torch.max(outputs, 1)
            
            all_predicts.extend(predicts.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
           
    
    validation_loss = running_loss / len(validation_loader)
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)
    
    return validation_loss, all_predicts, all_labels

if __name__ == "__main__":
    if torch.cuda.is_available():
        # use cuda
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    torch.manual_seed(1337)
    np.random.seed(1337)

    print(f"Device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    validation_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    train_dir = './archive/DATASET/TRAIN'
    validation_dir = './archive/DATASET/TEST'

    print("Loading data...")
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform = validation_transforms)

    print(f"Trains: {len(train_dataset)}")
    print(f"Validations: {len(validation_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = 128, shuffle = False)

    # Compute and print class distribution
    train_targets = np.array(train_dataset.targets)
    class_distribution = np.bincount(train_targets)
    print("\nClass distribution in training set:")
    for i, count in enumerate(class_distribution):
        print(f"Class {train_dataset.classes[i]}: {count} samples ({count/len(train_dataset)*100:.2f}%)")

    # Compute Class Weights
    class_weights = compute_class_weight('balanced', classes = np.unique(train_targets), y = train_targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initilize mobilenet_v2 model for binary classification
    print("Initializing mobilenet_v2 model for binary classification...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(in_features = model.classifier[1].in_features, out_features = 2)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Training Loop
    epochs = 50

    print("Start training...")
    training_start = time()

    for epoch in range(epochs):
        epoch_start = time() 
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        # Evaluate
        validation_loss, validation_predicts, validation_labels = evaluate_model(model, validation_loader, criterion)
        
        epoch_end = time()
        epoch_duration = epoch_end - epoch_start
        # Calculate metrics
        accuracy = (validation_predicts == validation_labels).mean() * 100
        precision = precision_score(validation_labels, validation_predicts, average='weighted')
        recall = recall_score(validation_labels, validation_predicts, average='weighted')
        f1 = f1_score(validation_labels, validation_predicts, average='weighted')
           
        print(f"Epoch {epoch+1} Complete:")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Epoch duration: {epoch_duration:.2f} seconds")
        

    training_duration = time() - training_start
    print(f"Training completed in {training_duration:.2f} seconds")
    accuracy = (validation_predicts == validation_labels).mean() * 100
    precision = precision_score(validation_labels, validation_predicts, average='weighted')
    recall = recall_score(validation_labels, validation_predicts, average='weighted')
    f1 = f1_score(validation_labels, validation_predicts, average='weighted')
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    # Final Evaluation
    print("Final Model Evaluation:")
    validation_loss, final_predicts, final_labels = evaluate_model(model, validation_loader, criterion)

    # Print detailed classification report
    print("Detailed Classification Report:")
    print(classification_report(final_labels, final_predicts, 
                            target_names=train_dataset.classes,
                            digits=4))
    torch.save(model.state_dict(), 'model.pth')
 