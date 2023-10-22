# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Load the data from the CSV file
# df = pd.read_csv('book1.csv')
#
# # Create a graph
# G = nx.Graph()
#
# # Extract x and y points from the dataframe
# x_points = df['Point of Regard Right X'].tolist()
# y_points = df['Point of Regard Right Y'].tolist()
#
# # Add nodes to the graph
# for i, (x, y) in enumerate(zip(x_points, y_points)):
#     G.add_node(i, x=x, y=y)
#
# # Add edges to the graph
# for i in range(len(x_points) - 1):
#     G.add_edge(i, i + 1)
#
# #draw graph
# pos = {i: (x, y) for i, (x, y) in enumerate(zip(x_points, y_points))}
# nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=50, edge_color="gray")
#
# plt.title('Eye Movement Graph')
# plt.xlabel('Point of Regard Right X')
# plt.ylabel('Point of Regard Right Y')
# plt.grid(True)
# plt.show()
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import random
from collections import Counter
import torch
from torch_geometric.data import DataLoader, Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


def create_graph_from_csv(file_name, label):
    df = pd.read_csv(file_name)
    x_points = df['Point of Regard Right X'].tolist()
    y_points = df['Point of Regard Right Y'].tolist()

    G = nx.Graph()
    # Adding nodes to the graph
    for i in range(len(x_points)):
        G.add_node(i, x=x_points[i], y=y_points[i])
    # Connecting consecutive nodes
    for i in range(len(x_points) - 1):
        G.add_edge(i, i + 1)
    # Assign label to the graph
    G.graph['label'] = label
    return G



file_names = ["book1.csv", "book2.csv", "book3.csv", "book4.csv", "book5.csv", "book6.csv", "book7.csv", "book8.csv",
              "book9.csv", "book10.csv", "book11.csv", "book12.csv"]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ,1]

graphs = [create_graph_from_csv(file_name, label) for file_name, label in zip(file_names, labels)]

# Plotting graphs
for idx, graph in enumerate(graphs, 1):
    plt.figure(idx)
    pos = {(node[0]): (node[1]['x'], node[1]['y']) for node in graph.nodes(data=True)}
    nx.draw(graph, pos, with_labels=True)
    title = "Autism" if graph.graph['label'] == 1 else "Normal"
    plt.title(f"Graph for {file_names[idx - 1]} - {title}")
    plt.show()


# Converting the NetworkX graphs into a format suitable for GNN libraries
def convert_to_pyg_format(graph):
    x = torch.tensor([[data['x'], data['y']] for _, data in graph.nodes(data=True)], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    y = torch.tensor([graph.graph['label']], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

pyg_graphs = [convert_to_pyg_format(graph) for graph in graphs]

# Model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(self.fc(x), dim=1)

model = GNNModel(input_dim=2, hidden_dim=64, output_dim=2)

# data shuffling
combined = list(zip(pyg_graphs, labels))
random.shuffle(combined)
pyg_graphs[:], labels[:] = zip(*combined)

# Data Spliting
train_data = pyg_graphs[:7]
test_data = pyg_graphs[7:]
train_labels = labels[:7]
test_labels = labels[7:]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())

    return correct / len(loader.dataset), all_preds, all_labels


num_epochs = 100
for epoch in range(num_epochs):
    loss = train()
    train_acc, _, _ = test(train_loader)
    test_acc, _, _ = test(test_loader)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

train_acc, train_preds, train_labels = test(train_loader)
test_acc, test_preds, test_labels = test(test_loader)

train_cm = confusion_matrix(train_labels, train_preds)
test_cm = confusion_matrix(test_labels, test_preds)
print("Train Confusion Matrix:")
print(train_cm)
print("Test Confusion Matrix:")
print(test_cm)
report=classification_report(test_labels, test_preds)
print(report)

train_labels_count = dict(Counter(train_labels))
test_labels_count = dict(Counter(test_labels))

print("Training data counts:", train_labels_count)
print("Testing data counts:", test_labels_count)
