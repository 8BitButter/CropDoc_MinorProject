# Required Libraries:
# pip install torch torchvision torchaudio scikit-learn numpy tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import copy
from torch.utils.tensorboard import SummaryWriter # Import TensorBoard writer

# --- Configuration ---

# !! Important: Update these paths !!
DATA_DIR = r"C:\Users\lenov\Downloads\data\tomato"
MODEL_SAVE_PATH = r"C:\Users\lenov\Downloads\models\pretrained\output"
BEST_OVERALL_MODEL_FILENAME = "tomato_best_model.pth"

# --- TensorBoard Log Directory ---
# Use an ABSOLUTE path to make it easier to find, or a relative path if you prefer.
# Make sure the parent directory (e.g., Desktop) exists and you have write permissions.
# TENSORBOARD_LOG_DIR = "runs/tomato_experiment" # Relative path example
TENSORBOARD_LOG_DIR = r"C:\Users\lenov\Desktop\Tomato_TensorBoard_Logs" # Absolute path example - CHANGE AS NEEDED

# Training Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4 # Adjust based on your system (e.g., 4)
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# --- Setup ---

def setup_device():
    """Sets up the device (GPU if available, otherwise CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_data_transforms():
    """Returns a dictionary of data transformations for train, validation, and test."""
    # (Same as before)
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def load_datasets(data_dir, data_transforms):
    """Loads train, validation, and test datasets."""
    # (Same as before)
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    class_names = None
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.isdir(test_dir):
        print(f"Warning: Test directory not found at {test_dir}. Testing will be skipped.")
        splits = ['train', 'valid']
    else:
        splits = ['train', 'valid', 'test']

    for x in splits:
        try:
            image_datasets[x] = datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            shuffle = (x == 'train')
            dataloaders[x] = DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)
            dataset_sizes[x] = len(image_datasets[x])
            print(f"Loaded {dataset_sizes[x]} images for {x}")
        except FileNotFoundError:
            print(f"Error: Dataset directory not found for split '{x}' at {os.path.join(data_dir, x)}")
            if x == 'train' or x == 'valid': raise
            else:
                 print(f"Skipping {x} split.")
                 if x in splits: splits.remove(x)

    if 'train' in image_datasets:
        class_names = image_datasets['train'].classes
        print(f"Classes: {class_names}")
    else:
        raise FileNotFoundError("Training data is required.")

    return image_datasets, dataloaders, dataset_sizes, class_names, splits


def modify_model_classifier(model_name, model, num_classes):
    """Modifies the classifier layer of a pre-trained model."""
    # (Same as before)
    for param in model.parameters(): param.requires_grad = False

    if model_name == "ResNet50":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        for param in model.fc.parameters(): param.requires_grad = True
    elif model_name == "AlexNet":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        for param in model.classifier[6].parameters(): param.requires_grad = True
    elif model_name == "MobileNet":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        for param in model.classifier[1].parameters(): param.requires_grad = True
    elif model_name == "DenseNet":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        for param in model.classifier.parameters(): param.requires_grad = True
    elif model_name == "EfficientNet":
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_features, num_classes))
        for param in model.classifier.parameters(): param.requires_grad = True
    else:
        print(f"Warning: Classifier modification logic not defined for {model_name}.")
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            for param in model.fc.parameters(): param.requires_grad = True
        elif hasattr(model, 'classifier'):
             if isinstance(model.classifier, nn.Sequential):
                 try:
                     last_layer = model.classifier[-1]
                     if isinstance(last_layer, nn.Linear):
                         in_features = last_layer.in_features
                         model.classifier[-1] = nn.Linear(in_features, num_classes)
                         for param in model.classifier[-1].parameters(): param.requires_grad = True
                     else: raise TypeError("Last classifier layer not Linear")
                 except Exception as e: print(f"Could not auto-modify Sequential classifier for {model_name}: {e}")
             elif isinstance(model.classifier, nn.Linear):
                 in_features = model.classifier.in_features
                 model.classifier = nn.Linear(in_features, num_classes)
                 for param in model.classifier.parameters(): param.requires_grad = True
             else: print(f"Unhandled classifier type {type(model.classifier)} for {model_name}")
        else: print(f"Cannot find 'fc' or 'classifier' for {model_name}")

    print(f"Modified {model_name} classifier for {num_classes} classes.")
    return model

# --- Training and Evaluation Functions ---

def train_and_evaluate_model(model, model_name_str, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs, model_save_dir, writer): # Added writer
    """Trains and validates the model, saving the best weights and logging to TensorBoard."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model_save_filename = os.path.join(model_save_dir, f"{model_name_str}_best_val.pth")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- TensorBoard Logging ---
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            # --- End TensorBoard Logging ---

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_save_filename)
                print(f"Saved new best model weights to {model_save_filename}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc


def test_model(model, dataloader, criterion, device, dataset_size):
    """Evaluates the model on the test set and returns metrics."""
    # (Same as before)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds, all_labels = [], []

    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / dataset_size
    test_acc = running_corrects.double() / dataset_size
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    print(f'Test Precision (Macro): {precision:.4f}')
    print(f'Test Recall (Macro): {recall:.4f}')
    print(f'Test F1-Score (Macro): {f1:.4f}')

    return test_loss, test_acc.item(), precision, recall, f1


# --- Main Execution ---

def main():
    device = setup_device()
    data_transforms = get_data_transforms()

    # --- Create Directories ---
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    try:
        # Attempt to create the base TensorBoard directory
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
        print(f"Base TensorBoard log directory: {os.path.abspath(TENSORBOARD_LOG_DIR)}")
    except OSError as e:
        print(f"Error creating TensorBoard directory '{TENSORBOARD_LOG_DIR}': {e}")
        print("Please check the path and permissions. Exiting.")
        return # Exit if we can't create the log directory

    image_datasets, dataloaders, dataset_sizes, class_names, available_splits = load_datasets(DATA_DIR, data_transforms)
    num_classes = len(class_names)

    # --- Define Models to Train ---
    models_to_train = {
        # "AlexNet": lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        # "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "MobileNet": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        # "DenseNet": lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        # "EfficientNet": lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1), # Example: training two models
    }

    model_performance = {}
    best_overall_acc = 0.0
    best_overall_model_name = None
    best_overall_model_log_dir = None # Store log dir path for the best model
    best_overall_model_weights_path = os.path.join(MODEL_SAVE_PATH, BEST_OVERALL_MODEL_FILENAME)

    # --- Training Loop ---
    for model_name, model_loader in models_to_train.items():
        print(f"\n--- Training {model_name} ---")

        # --- TensorBoard Writer Setup ---
        current_run_time = int(time.time())
        # Create a unique directory for this specific run inside the base log directory
        log_dir = os.path.join(TENSORBOARD_LOG_DIR, f"{model_name}_{current_run_time}")
        try:
             writer = SummaryWriter(log_dir=log_dir)
             # Print the ABSOLUTE path to remove ambiguity
             print(f"TensorBoard log directory (absolute): {os.path.abspath(log_dir)}")
        except Exception as e:
             print(f"Error creating SummaryWriter for '{log_dir}': {e}")
             print("Skipping TensorBoard logging for this run.")
             writer = None # Set writer to None so logging calls are skipped
        # --- End TensorBoard Setup ---

        model = model_loader()
        model = modify_model_classifier(model_name, model, num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

        # Train and evaluate, passing the writer
        if writer: # Only train if writer was created successfully
            trained_model, best_val_acc = train_and_evaluate_model(
                model, model_name, criterion, optimizer, dataloaders, dataset_sizes,
                device, NUM_EPOCHS, MODEL_SAVE_PATH, writer
            )
        else: # Handle case where writer failed
             print("Skipping training for this run due to TensorBoard writer error.")
             continue # Go to the next model

        # --- Log Hyperparameters and Final Metric (only if writer exists) ---
        if writer:
            hparams = {
                'model': model_name,
                'lr': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'optimizer': optimizer.__class__.__name__,
                'epochs': NUM_EPOCHS
            }
            final_metrics = {
                'hparam/best_val_accuracy': best_val_acc.item()
            }
            try:
                 writer.add_hparams(hparams, final_metrics)
            except Exception as e:
                 print(f"Error logging HParams: {e}")
            writer.close() # Close writer to flush logs
        # --- End HParam Logging ---

        model_performance[model_name] = {
            "best_val_acc": best_val_acc.item(),
            "log_dir": log_dir # Store log directory path
        }

        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_overall_model_name = model_name
            best_overall_model_log_dir = log_dir # Keep track of the best model's log dir
            torch.save(trained_model.state_dict(), best_overall_model_weights_path)
            print(f"*** New best overall model: {model_name} (Val Acc: {best_overall_acc:.4f}). Saved to {best_overall_model_weights_path} ***")

    # --- Final Summary and Testing ---
    print("\n--- Training Summary ---")
    sorted_performance = sorted(model_performance.items(), key=lambda item: item[1]['best_val_acc'], reverse=True)
    for model_name, metrics in sorted_performance:
        # Print absolute path here too for clarity
        print(f"{model_name}: Best Validation Accuracy = {metrics['best_val_acc']:.4f} (Log Dir: {os.path.abspath(metrics['log_dir'])})")

    if best_overall_model_name:
        print(f"\nBest overall model (based on validation): {best_overall_model_name} with accuracy {best_overall_acc:.4f}")

        if 'test' in available_splits:
            print(f"\n--- Testing the best model: {best_overall_model_name} ---")
            best_model_arch = models_to_train[best_overall_model_name]()
            best_model_arch = modify_model_classifier(best_overall_model_name, best_model_arch, num_classes)
            try:
                map_location = None if torch.cuda.is_available() else torch.device('cpu')
                best_model_arch.load_state_dict(torch.load(best_overall_model_weights_path, map_location=map_location))
                print(f"Successfully loaded weights from {best_overall_model_weights_path}")
            except Exception as e:
                 print(f"Error loading weights for the best model: {e}. Skipping testing.")
                 return

            best_model_arch = best_model_arch.to(device)
            criterion = nn.CrossEntropyLoss()

            test_loss, test_acc, precision, recall, f1 = test_model(best_model_arch, dataloaders['test'], criterion, device, dataset_sizes['test'])

            # --- Log Test Metrics to TensorBoard ---
            if best_overall_model_log_dir:
                try:
                     # Re-open writer for the best model's specific run directory
                     test_writer = SummaryWriter(log_dir=best_overall_model_log_dir)
                     test_writer.add_scalar('Metrics/Test_Accuracy', test_acc, 0)
                     test_writer.add_scalar('Metrics/Test_Loss', test_loss, 0)
                     test_writer.add_scalar('Metrics/Test_Precision_Macro', precision, 0)
                     test_writer.add_scalar('Metrics/Test_Recall_Macro', recall, 0)
                     test_writer.add_scalar('Metrics/Test_F1_Macro', f1, 0)
                     test_writer.close()
                     print(f"Logged test metrics to TensorBoard: {os.path.abspath(best_overall_model_log_dir)}")
                except Exception as e:
                     print(f"Error creating SummaryWriter or logging test metrics for '{best_overall_model_log_dir}': {e}")
            else:
                print("Warning: Could not determine log directory for the best model to log test metrics.")
            # --- End Test Metric Logging ---

        else:
            print("\nTest dataset not available or failed to load. Skipping final testing.")
    else:
        print("\nNo models were trained successfully or no best model was identified.")

    # --- Final TensorBoard Instructions ---
    print("\n\n===================== TensorBoard Usage =====================")
    print(f"Log files were saved under the base directory:")
    print(f"  {os.path.abspath(TENSORBOARD_LOG_DIR)}")
    print("\nTo view the results, open a NEW terminal/command prompt and run:")
    print(f"  tensorboard --logdir \"{os.path.abspath(TENSORBOARD_LOG_DIR)}\"")
    print("\nThen open the URL (usually http://localhost:6006/) in your browser.")
    print("=============================================================")


if __name__ == "__main__":
    main()