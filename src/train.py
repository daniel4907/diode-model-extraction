import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 

from src.models import *
from src.utils import *

# Generate diode I-V training data

x, y = generate_training_data_diode(n_samples=10000, v_range=(0, 1.0))
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

split_idx = int(0.8 * len(x_tensor)) # 80% train, 20% validation
train_x, val_x = x_tensor[:split_idx], x_tensor[split_idx:]
train_y, val_y = y_tensor[:split_idx], y_tensor[split_idx:]

model = DiodeNet(input_size=150)
crit = nn.MSELoss() # (prediction - target)^2
optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate of 0.001

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    preds = model(train_x)
    loss = crit(preds, train_y)
    
    optimizer.zero_grad() # clear gradients
    loss.backward() # find new gradients
    optimizer.step() # update weights
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = crit(val_preds, val_y)
        print(f"Epoch {epoch}: train loss = {loss.item():.6f}, val loss = {val_loss.item():.6f}")
        
torch.save(model.state_dict(), 'models/diode_model_weights.pth')

# Generate diode C-V training data

x, y = generate_training_data_cv_diode(n_samples=10000)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

split_idx = int(0.8 * len(x_tensor)) # 80% train, 20% validation
train_x, val_x = x_tensor[:split_idx], x_tensor[split_idx:]
train_y, val_y = y_tensor[:split_idx], y_tensor[split_idx:]

model = DiodeNet(input_size=150)
crit = nn.MSELoss() # (prediction - target)^2
optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate of 0.001

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    preds = model(train_x)
    loss = crit(preds, train_y)
    
    optimizer.zero_grad() # clear gradients
    loss.backward() # find new gradients
    optimizer.step() # update weights
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = crit(val_preds, val_y)
        print(f"Epoch {epoch}: train loss = {loss.item():.6f}, val loss = {val_loss.item():.6f}")
        
torch.save(model.state_dict(), 'models/diode_cv_model_weights.pth')

# Generate MOSFET transfer characteristics training data

x, y = generate_training_transfer_mosfet(n_samples=10000)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

split_idx = int(0.8 * len(x_tensor)) # 80% train, 20% validation
train_x, val_x = x_tensor[:split_idx], x_tensor[split_idx:]
train_y, val_y = y_tensor[:split_idx], y_tensor[split_idx:]

model = MOSFETNet(input_size=152)
crit = nn.MSELoss() # (prediction - target)^2
optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate of 0.001

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    preds = model(train_x)
    loss = crit(preds, train_y)
    
    optimizer.zero_grad() # clear gradients
    loss.backward() # find new gradients
    optimizer.step() # update weights
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = crit(val_preds, val_y)
        print(f"Epoch {epoch}: train loss = {loss.item():.6f}, val loss = {val_loss.item():.6f}")
        
torch.save(model.state_dict(), 'models/mosfet_transfer_model_weights.pth')

# Generate MOSFET output characteristics training data

x, y = generate_training_output_mosfet(n_samples=10000)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

split_idx = int(0.8 * len(x_tensor)) # 80% train, 20% validation
train_x, val_x = x_tensor[:split_idx], x_tensor[split_idx:]
train_y, val_y = y_tensor[:split_idx], y_tensor[split_idx:]

model = MOSFETNet(input_size=152)
crit = nn.MSELoss() # (prediction - target)^2
optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate of 0.001

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    preds = model(train_x)
    loss = crit(preds, train_y)
    
    optimizer.zero_grad() # clear gradients
    loss.backward() # find new gradients
    optimizer.step() # update weights
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = crit(val_preds, val_y)
        print(f"Epoch {epoch}: train loss = {loss.item():.6f}, val loss = {val_loss.item():.6f}")
        
torch.save(model.state_dict(), 'models/mosfet_output_model_weights.pth')