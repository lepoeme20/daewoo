import torch
import torch.nn as nn
import torch.nn.functional as F
from build_dataloader import get_dataloader
import numpy as np
import random


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
csv_path = '../data/weather_data_label.csv'
root_img_path = '/media/heejeong/HDD1/daewoo/data/weather/'
save_path = 'model_Adam.pt'
csv_save_path = 'result_Adam.csv'

ker_size = 9
img_size = 32
P = 3
Q = 3
C = 1
num_epochs = 100
batch_size = 40000
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.9
seed = 42

# Seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

torch.utils.backcompat.broadcast_warning.enabled = True

# Data loader
trn_loader, dev_loader, tst_loader = get_dataloader(
    csv_path=csv_path,
    root_img_path=root_img_path,
    batch_size=batch_size,
    img_size=img_size,
    P=P,
    Q=Q,
    C=C
)

def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y)

# Model
class CNN(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        q = self.fc3(h)

        return q

model = CNN(ker_size=ker_size, n_kers=50, n1_nodes=800, n2_nodes=800)
model = model.to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


# Train the model
total_step = len(trn_loader)
best_loss = 10000
for epoch in range(num_epochs):
    # Train
    model.train()
    for i, (images, labels) in enumerate(trn_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs.squeeze(1), labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, labels in dev_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(1), labels)
            val_loss += loss.item()
            
        print ('Epoch [{}/{}], Valdation Loss: {:.4f}' 
            .format(epoch+1, num_epochs, val_loss/len(dev_loader)))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)


# Test the model
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def return_perf(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, mape

model.load_state_dict(torch.load(save_path))

trues = np.array([])
preds = np.array([])

model.eval()
with torch.no_grad():
    for images, labels in tst_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        trues = np.r_[trues, labels.detach().cpu().numpy()]
        preds = np.r_[preds, outputs.detach().cpu().numpy().squeeze(1)]
    
    mse, mae, mape = return_perf(trues, preds)

    print('Test MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(mse, mae, mape))

result = pd.DataFrame()
result['true'] = list(trues)
result['pred'] = list(preds)
result.to_csv(csv_save_path, index=False)
