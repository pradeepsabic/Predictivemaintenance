import torch
 #Predictions and Insights-Create a function for making predictions on new data
def predict(model, new_data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(new_data.values, dtype=torch.float32)
        return (model(inputs).numpy() > 0.5).astype(int)
