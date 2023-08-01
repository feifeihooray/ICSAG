import torch
from torch import optim
from torch.nn import BCELoss
from model_build import LSTMModel
from data_setup import load_data
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

def train_model():
    X_train, X_test, y_train, y_test, train_loader, test_loader = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(train_loader.dataset.tensors[0].shape[2]).to(device)

    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in tqdm(range(10), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total
            print(f'Epoch {epoch}, loss: {running_loss/len(train_loader)}, val_loss: {val_loss/len(test_loader)}, accuracy: {accuracy}')




# def train_model():
#     X_train, X_test, y_train, y_test = load_data()
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = LSTMModel(X_train.shape[2]).to(device)
#
#     criterion = BCELoss()
#     optimizer = optim.Adam(model.parameters())
#
#     # Set accumulation steps
#     accumulation_steps = 10  # Change this to a suitable number
#
#     X_train = X_train.astype(float)
#     y_train = y_train.astype(float)
#     X_test = X_test.astype(float)
#     y_test = y_test.astype(float)
#
#     X_train = torch.tensor(X_train).float().to(device)
#     X_test = torch.tensor(X_test).float().to(device)
#     y_train = torch.tensor(y_train).float().unsqueeze(1).to(device)
#     y_test = torch.tensor(y_test).float().unsqueeze(1).to(device)
#
#     for epoch in tqdm(range(10), desc="Training Progress"):
#         model.train()
#         running_loss = 0.0
#         for i in range(len(X_train)):
#             optimizer.zero_grad()  # We need to set the gradients to zero at the beginning of each accumulation cycle
#             outputs = model(X_train[i].unsqueeze(0))
#             loss = criterion(outputs, y_train[i].unsqueeze(0))
#             loss.backward()  # Backpropagate the loss
#             running_loss += loss.item()
#
#             # Check if it's time to update the model parameters
#             if (i+1) % accumulation_steps == 0:
#                 optimizer.step()  # Update the model parameters
#                 optimizer.zero_grad()  # Clear the gradients
#
#                 model.eval()
#                 with torch.no_grad():
#                     val_outputs = model(X_test)
#                     val_loss = criterion(val_outputs, y_test)
#                     if epoch % 10 == 0:
#                         y_test_preds = torch.round(val_outputs)
#                         val_accuracy = (y_test_preds == y_test).float().mean()
#
#                 print(f'Epoch {epoch}, Step {i+1}, loss: {running_loss / accumulation_steps}, val_loss: {val_loss.item()}, val_accuracy: {val_accuracy.item()}')
#                 model.train()
#
#         # Outside the accumulation cycle, remember to perform the update one last time
#         if len(X_train) % accumulation_steps != 0:
#             optimizer.step()  # Update the model parameters
#             optimizer.zero_grad()  # Clear the gradients
#
#             model.eval()
#             with torch.no_grad():
#                 val_outputs = model(X_test)
#                 val_loss = criterion(val_outputs, y_test)
#                 if epoch % 10 == 0:
#                     y_test_preds = torch.round(val_outputs)
#                     val_accuracy = (y_test_preds == y_test).float().mean()
#
#             print(f'Epoch {epoch}, loss: {running_loss / accumulation_steps}, val_loss: {val_loss.item()}, val_accuracy: {val_accuracy.item()}')
#
#     # Save the model
#     if not os.path.exists('model'):
#         os.makedirs('model')
#     torch.save(model.state_dict(), 'model/model.ckpt')
#     print('Model saved to model/model.ckpt')


if __name__ == '__main__':
    train_model()
