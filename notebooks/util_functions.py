from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import json

def compute_metrics(true_labels: list,
                    predicted_labels: list) -> dict:
    """
    Computes classification eval metrics

    Args:
    true_labels: list
        The ground truth labels
    predicted_labels: list
        The predicted labels
    Returns:
    metrics_dict: dict
        The classification metrics
    """
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    metrics_dict = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4)
    }
    return metrics_dict


def print_eval_metrics(metrics: dict) -> None:
    """Prints the Classification eval metrics"""
    print(f"The precision score is:  {metrics['precision']:.2f}")
    print(f"The recall score is:  {metrics['recall']:.2f}")
    print(f"The f1_score score is:  {metrics['f1_score']:.2f}")
    print(f"The accuracy score is:  {metrics['accuracy']:.2f}")
    pass


def save_metrics(metrics: dict, file_name: str = "metrics") -> None:
    """Saves metrics in a .json in current directory"""
    with open(f"./{file_name}.json", "w") as file:
        json.dump(metrics, file)
    print(f"Saved {file_name}.json!")
    pass
