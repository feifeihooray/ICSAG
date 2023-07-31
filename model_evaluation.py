import torch
from sklearn.metrics import accuracy_score
from data_setup import load_data
from model_build import LSTMModel


def evaluate_model():
    # Load the data
    _, _, test_X, test_y = load_data('data/Physical/SWaT_Dataset_Attack_v0.xlsx')

    # Define the model
    model = LSTMModel(input_size=test_X.size(1), hidden_size=64, num_layers=2, num_classes=2)

    # Load the trained model
    model.load_state_dict(torch.load('model.ckpt'))

    # Test the model
    outputs = model(test_X)
    _, predicted = torch.max(outputs.data, 1)

    # Calculate the accuracy
    accuracy = accuracy_score(test_y.numpy(), predicted.numpy())
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
