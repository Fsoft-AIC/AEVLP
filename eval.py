import torch 
import time 
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
import json
import datasets
import models
import os 
from models import ImageClassifier

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0  # type: ignore
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

def get_configs(path):
    config_path = os.path.join(path, "config.json")
    # read config file
    with open(config_path, 'r') as f:
        P = json.load(f)
    return P
def get_model(path, P):
    checkpoint_path = os.path.join(path, "bestmodel.pt")
    model = ImageClassifier(P)
    # checkpoint path
    model_state, _ = torch.load(checkpoint_path)
    model.load_state_dict(model_state)
    return model

# Function to compute evaluation metrics (F1-score, precision, recall, accuracy) for multiple thresholds
def compute_metrics(y_true, y_pred, mode='micro', thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Compute evaluation metrics (F1-score, precision, recall, accuracy) for multiple thresholds.

    Parameters:
    y_true (numpy array): True labels, shape (num_examples, num_classes).
    y_pred (numpy array): Predicted probabilities, shape (num_examples, num_classes).
    mode (str): 'micro' for micro-averaging, 'macro' for macro-averaging.
    thresholds (list): List of thresholds for converting probabilities to binary predictions.

    Returns:
    pandas DataFrame: DataFrame containing computed metrics for each threshold.
    """
    metrics_df = pd.DataFrame(index=['F1 Score', 'Precision', 'Recall', 'Accuracy'])
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the threshold
        y_pred_binary = (y_pred > threshold).astype(int)

        if mode == 'micro':
            # Compute micro-averaged metrics
            f1 = f1_score(y_true.ravel(), y_pred_binary.ravel())
            precision = precision_score(y_true.ravel(), y_pred_binary.ravel())
            recall = recall_score(y_true.ravel(), y_pred_binary.ravel())
            accuracy = accuracy_score(y_true.ravel(), y_pred_binary.ravel())

        elif mode == 'macro':
            # Compute macro-averaged metrics
            f1 = f1_score(y_true, y_pred_binary, average='macro')
            precision = precision_score(y_true, y_pred_binary, average='macro')
            recall = recall_score(y_true, y_pred_binary, average='macro')

            num_classes = y_true.shape[1]
            accuracies = []
            for i in range(num_classes):
                true_labels_i = y_true[:, i]
                pred_labels_i = y_pred_binary[:, i]
                accuracy_i = np.mean(true_labels_i == pred_labels_i)
                accuracies.append(accuracy_i)

            accuracy = np.mean(accuracies)
        else:
            raise ValueError("Invalid mode. Use 'micro' or 'macro'.")

        metrics_df[threshold] = [f1 * 100, precision * 100, recall * 100, accuracy * 100]
        # round to 2 decimal places
        metrics_df = metrics_df.round(2)
        print(f"Threshold: {threshold}, F1 Score: {f1 * 100:.2f}, Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}, Accuracy: {accuracy * 100:.2f}")
    return metrics_df

# Function to compute metrics for each class and save to Excel
def compute_and_save_metrics_for_each_class(y_true, y_pred, class_names):
    """
    Compute evaluation metrics (mAP, precision, recall) for each class and save to Excel.

    Parameters:
    y_true (numpy array): True labels, shape (num_examples, num_classes).
    y_pred (numpy array): Predicted probabilities, shape (num_examples, num_classes).
    class_names (list): List of class names (length num_classes).
    excel_filename (str): File name to save the results in Excel.

    Returns:
    None
    """
    # Get unique thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 1.0, 0.1)

    # Create a DataFrame to store the metrics
    precision_names = [f'precision_{t:.1f}' for t in thresholds]
    recall_names = [f'recall_{t:.1f}' for t in thresholds]
    row_names = [] 
    for i in range(len(precision_names)):
        row_names.append(precision_names[i])
        row_names.append(recall_names[i])
    metrics_df = pd.DataFrame(index=['mAP'] + row_names)

    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]

        # Compute average precision
        ap = average_precision_score(y_true_class, y_pred_class) * 100

        # Compute precision and recall for each threshold
        precisions = [precision_score(y_true_class, (y_pred_class > t).astype(int)) * 100 for t in thresholds]
        recalls = [recall_score(y_true_class, (y_pred_class > t).astype(int)) * 100 for t in thresholds]

        precision_recall = []
        for i in range(len(precisions)):
            precision_recall.append(precisions[i])
            precision_recall.append(recalls[i])
        # Add the metrics to the DataFrame
        metrics_df[class_name] = [ap] + precision_recall
     
        # round to 2 decimal places
        metrics_df = metrics_df.round(2)

    return metrics_df

def compute_micro_map(y_true, y_pred_probs):
    """
    Compute micro mAP (mean Average Precision) for multi-class classification.

    Parameters:
    y_true (numpy array): True labels, shape (num_examples, num_classes).
    y_pred_probs (numpy array): Predicted probabilities, shape (num_examples, num_classes).

    Returns:
    float: Micro mAP score.
    """
    # Flatten the true labels and predicted probabilities
    y_true_flattened = y_true.ravel()
    y_pred_probs_flattened = y_pred_probs.ravel()

    # Compute the average precision
    micro_map = average_precision_score(y_true_flattened, y_pred_probs_flattened)

    return micro_map


# Function to save the metrics to an Excel file
def evaluate(y_true, y_pred, output='./metrics'):
    # with open('data/glacemood/moods_labelname.json', 'r') as f:
    #     class_names_dict = json.load(f)
    # class_names = [str(k) + ' - ' + v for k, v in class_names_dict.items()]
    # Compute metrics using the compute_metrics function
    # metrics_micro_df = compute_metrics(y_true, y_pred, 'micro')
    metrics_macro_df = compute_metrics(y_true, y_pred, 'macro', [0.5])
    
    # Compute metrics by class 
    # metrics_by_class_df = compute_and_save_metrics_for_each_class(y_true, y_pred, class_names)

    # Save to Excel
    # metrics_micro_df.to_excel(f'{output}/metrics_micro.xlsx')
    metrics_macro_df.to_excel(f'{output}/metrics_macro.xlsx')
    # metrics_by_class_df.to_excel(f'{output}/metrics_by_class.xlsx')
    
    print("Saved metrics to folder: ", output)


def test(path) -> None:
    
    P = get_configs(path)
    
    dataset = datasets.get_data(P)
    
    dataloader = torch.utils.data.DataLoader(
        dataset['test'],
        batch_size = P['bsize']*5,
        shuffle = False,
        sampler = None,
        num_workers = P['num_workers'],
        drop_last = False,
        pin_memory = True
    )

    model = get_model(path, P)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    y_pred = np.zeros((len(dataset['test']), P['num_classes']))
    y_true = np.zeros((len(dataset['test']), P['num_classes']))
    batch_stack = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']


            logits = model(image)
            
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
               
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size
        
    macro_mAP_score = mAP(y_true, y_pred)
    # micro_mAP_score = 100 * compute_micro_map(y_true, y_pred)
    print(f"macro mAP score: {macro_mAP_score}")
    # print(f"micro mAP score: {micro_mAP_score}") 
    
    # # Save to text file
    # with open("./metrics/mAP.txt", "w") as f:
    #     f.write(f"macro mAP score: {macro_mAP_score}\n")
    #     f.write(f"micro mAP score: {micro_mAP_score}\n") 
    
    # Save to Excel
    evaluate(y_true, y_pred)
    
    print("Finish evaluation!")
    
if __name__ == '__main__':
    path = "/home/s/luongtk/GRLoss/results/20240629_183550"
    print(path)
    test(path)