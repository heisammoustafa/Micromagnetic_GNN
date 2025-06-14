# data_loader.py

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.utils.data import random_split 
from sklearn.preprocessing import StandardScaler
import yaml


# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
batch_size = config["batch_size"]
desired_feature_indices = config["desired_feature_indices"]
# Load data (same as before)

data_x = []
data_y = []
edge_index_raw = []

with open(f"PKL/data_x_edge_length_var.pkl", "rb") as file:
    data_x_trial = pickle.load(file)
    data_x.extend(data_x_trial)
    
with open(f"PKL/data_y_edge_length_var.pkl", "rb") as file:
    data_y_trial = pickle.load(file)
    data_y.extend(data_y_trial)
    
with open(f"PKL/edge_index_edge_length_var.pkl", "rb") as file:
    edge_index_raw_trial = pickle.load(file)
    edge_index_raw.extend(edge_index_raw_trial)
    
for i in range(1, 4):
    with open(f"PKL/data_x_{i}_V3.pkl", "rb") as file:
        data_x_trial = pickle.load(file)
        data_x.extend(data_x_trial)
        
    with open(f"PKL/data_y_{i}_V3.pkl", "rb") as file:
        data_y_trial = pickle.load(file)
        data_y.extend(data_y_trial)
        
    with open(f"PKL/edge_index_{i}_V3.pkl", "rb") as file:
        edge_index_raw_trial = pickle.load(file)
        edge_index_raw.extend(edge_index_raw_trial)

# Load _bigger datasets
with open(f"PKL/data_x_bigger.pkl", "rb") as file:
    data_x_big = pickle.load(file) 
    data_x.extend(data_x_big)

with open(f"PKL/data_y_bigger.pkl", "rb") as file:
    data_y_big = pickle.load(file)
    data_y.extend(data_y_big)

with open(f"PKL/edge_index_bigger.pkl", "rb") as file:
    edge_index_raw_big = pickle.load(file)
    edge_index_raw.extend(edge_index_raw_big)

# Load _bigger datasets
with open(f"PKL/data_x_bigger2.pkl", "rb") as file:
    data_x_big = pickle.load(file)
    data_x.extend(data_x_big)

with open(f"PKL/data_y_bigger2.pkl", "rb") as file:
    data_y_big = pickle.load(file)
    data_y.extend(data_y_big)

with open(f"PKL/edge_index_bigger2.pkl", "rb") as file:
    edge_index_raw_big = pickle.load(file)
    edge_index_raw.extend(edge_index_raw_big)
    
# Load _bigger datasets
with open(f"PKL/data_x_fisch_2.pkl", "rb") as file:
    data_x_big = pickle.load(file)
    for graph in range(len(data_x_big)):
        for grain in range(len(data_x_big[graph])):
            data_x_big[graph][grain][0] =  data_x_big[graph][grain][0]/  data_x_big[graph][grain][14]
            data_x_big[graph][grain][1] =  data_x_big[graph][grain][1]/  data_x_big[graph][grain][14]
            data_x_big[graph][grain][2] =  data_x_big[graph][grain][2]/  data_x_big[graph][grain][14]
            data_x_big[graph][grain][3] =  data_x_big[graph][grain][3]/  data_x_big[graph][grain][14]**3
            data_x_big[graph][grain][4] =  data_x_big[graph][grain][4]/  data_x_big[graph][grain][14]
            data_x_big[graph][grain][7] =  data_x_big[graph][grain][7]/  data_x_big[graph][grain][14]**2
    data_x.extend(data_x_big)

with open(f"PKL/data_y_fisch_2.pkl", "rb") as file:
    data_y_big = pickle.load(file)
    data_y.extend(data_y_big)

with open(f"PKL/edge_index_fisch_2.pkl", "rb") as file:
    edge_index_raw_big = pickle.load(file)
    edge_index_raw.extend(edge_index_raw_big)

for graph_features in data_x:
    for node_features in graph_features:
        node_features[0] = node_features[0]*node_features[14]
        node_features[1] = node_features[1]*node_features[14]
        node_features[2] = node_features[2]*node_features[14]
        node_features[3] = node_features[3]*node_features[14]**3
        node_features[4] = node_features[4]*node_features[14]
        node_features[7] = node_features[7]*node_features[14]**2
        node_features.extend([np.linalg.norm([node_features[0]-node_features[14]/2,
              node_features[1]-node_features[14]/2,node_features[2]-node_features[14]/2])])

indices_to_delete = []
# Loop through the first dimension of data_x
for i in range(len(data_x)):
     if len(data_x[i]) < 2:
        indices_to_delete.append(i)

for graph_features in data_x:
    for node_features in graph_features:
      node_features[:] = [node_features[i] for i in desired_feature_indices]
      
data_x_standardized = data_x
graphs = []

for i in range(len(data_x_standardized)):
    node_features = np.array(data_x_standardized[i])
    edge_index = torch.tensor(edge_index_raw[i], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[node_features[edge[0]][7]] for edge in edge_index_raw[i]], dtype=torch.float)
    node_features = np.delete(node_features, 7, axis=1) 
    label = torch.tensor(data_y[i], dtype=torch.float)
    data = Data(x=torch.tensor(node_features, dtype=torch.float), 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=label)
    graphs.append(data)

# Create PyG dataset
dataset = graphs

indices = [i for i in range(len(dataset)) if i not in indices_to_delete]
dataset = [dataset[index] for index in indices]
# Calculate the lengths for each subset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size  # Ensure the sizes add up to the full dataset

# Perform the random split
dataset_tr, dataset_val, dataset_te = random_split(dataset, [train_size, val_size, test_size])
# Save datasets
torch.save(dataset_tr, "dataset_tr.pt")
torch.save(dataset_val, "dataset_val.pt")
torch.save(dataset_te, "dataset_te.pt")

# Load datasets later
# dataset_tr = torch.load("dataset_tr.pt")
# dataset_val = torch.load("dataset_val.pt")
# dataset_te = torch.load("dataset_te.pt")
# Fitting on Training Data, then scaling ALL Data, including labels
# Collect all node features into a single array
all_node_features_train = []
for graph in dataset_tr:  # Assume this is the training set
    all_node_features_train.extend(graph.x.tolist())
all_node_features_train = np.array(all_node_features_train)

# Fit the scaler
scaler = StandardScaler() # feature scaler
label_scaler = StandardScaler()
edge_attr_scaler = StandardScaler()
scaler.fit(all_node_features_train)

# Apply transformation back to each graph
all_scaled_node_features_train = []
for graph in dataset_tr:
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float)
    all_scaled_node_features_train.extend(graph.x.tolist())
all_scaled_node_features_train = np.array(all_scaled_node_features_train)

for graph in dataset_val:  # Validation set
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float)
    
for graph in dataset_te:  # Validation set
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float)

# Collect all labels into a single array
all_labels_train = []
for graph in dataset_tr:  # Training set
    all_labels_train.append(graph.y.tolist())
all_labels_train = np.array(all_labels_train).reshape(-1, 1)  # Reshape for sklearn scaler

# Fit the scaler on training labels
label_scaler.fit(all_labels_train)

# Apply transformation back to each graph
for graph in dataset_tr:
    graph.y = torch.tensor(label_scaler.transform(graph.y.reshape(-1, 1)), dtype=torch.float)

for graph in dataset_val:  # Validation set
    graph.y = torch.tensor(label_scaler.transform(graph.y.reshape(-1, 1)), dtype=torch.float)

for graph in dataset_te:  # Test set
    graph.y = torch.tensor(label_scaler.transform(graph.y.reshape(-1, 1)), dtype=torch.float)

# Collect all edge attributes into a single array
all_edge_attr_train = []
for graph in dataset_tr:  # Training set
    if graph.edge_attr is not None:
        all_edge_attr_train.extend(graph.edge_attr.tolist())
all_edge_attr_train = np.array(all_edge_attr_train)

# Fit the scaler on training edge attributes
edge_attr_scaler.fit(all_edge_attr_train)

# Apply transformation back to each graph
for graph in dataset_tr:
    if graph.edge_attr is not None:
        graph.edge_attr = torch.tensor(edge_attr_scaler.transform(graph.edge_attr), dtype=torch.float)

for graph in dataset_val:  # Validation set
    if graph.edge_attr is not None:
        graph.edge_attr = torch.tensor(edge_attr_scaler.transform(graph.edge_attr), dtype=torch.float)

for graph in dataset_te:  # Test set
    if graph.edge_attr is not None:
        graph.edge_attr = torch.tensor(edge_attr_scaler.transform(graph.edge_attr), dtype=torch.float)

# DataLoader
train_loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)
test_loader = DataLoader(dataset_te, batch_size=batch_size)

# Export variables
__all__ = ['dataset_tr', 'dataset_val', 'dataset_te', 'train_loader', 'val_loader', 'test_loader', 'label_scaler']
