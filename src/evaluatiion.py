from sklearn.metrics import classification_report
import torch

#Model Evaluation- Performance Metrics-Evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test.values, dtype=torch.float32)
        outputs = model(inputs)
        predictions = (outputs.numpy() > 0.5).astype(int)  # Convert probabilities to binary predictions
    report = classification_report(y_test, predictions)
    print(report)
