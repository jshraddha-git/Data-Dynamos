import json
def accuracy_score(y_true, y_pred):
    """Return the fraction of predictions that match the true labels."""
    if not y_true:
        return 0.0
    return sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def classification_report(y_true, y_pred):
    """Return a compact classification report without scikit-learn."""
    labels = sorted(set(y_true) | set(y_pred))
    lines = ["              precision    recall  f1-score   support"]

    for label in labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(y_true, y_pred)
        )
        predicted_total = sum(predicted == label for predicted in y_pred)
        actual_total = sum(actual == label for actual in y_true)
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        lines.append(
            f"{label:>14} {precision:>9.2f} {recall:>9.2f} "
            f"{f1_score:>9.2f} {actual_total:>9}"
        )

    return "\n".join(lines)
# Import your model loader / inference functions from main.py or model module
from main import evaluate_text # Adjust based on your script structure

# Load your local training / test dataset
def load_test_data(filepath="data/test_dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    dataset = load_test_data()
    y_true = []
    y_pred = []

    print("Running evaluation against test/training samples...\n")
    for sample in dataset:
        text = sample["text"]
        true_label = sample["label"] # e.g., "blur", "neutralize", or "show"
        
        # Run inference through your backend logic
        result = evaluate_text(text=text, user_triggers=sample.get("triggers", []))
        predicted_action = result.get("action", "show")
        
        y_true.append(true_label)
        y_pred.append(predicted_action)

    print("--- Classification Metrics ---")
    print(classification_report(y_true, y_pred))
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    run_evaluation()