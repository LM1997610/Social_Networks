
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv # GraphConv, SplineConv

class GConv_Network(torch.nn.Module):

    def __init__(self, dataset):
        super().__init__()

        torch.manual_seed(42)
        self.dataset = dataset

        self.conv1 = GCNConv(dataset.num_features, 256)
        self.conv2 = GCNConv(256, 16)
        self.out = nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.softmax(self.out(x), dim=1)
        return x
    
class GAT_Network(nn.Module):
  
    def __init__(self, dataset):
        super().__init__()

        torch.manual_seed(42)
        self.dataset = dataset
        
        self.conv1 = GATConv(dataset.num_features, 256)
        self.conv2 = GATConv(256, 16)
        self.out = nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)
 
        x = F.softmax(self.out(x), dim=1)
        return x

class SAGE_Network(torch.nn.Module):

    def __init__(self, dataset):
        super().__init__()

        torch.manual_seed(42)
        self.dataset = dataset

        self.conv1 = SAGEConv(dataset.num_features, 256)
        self.conv2 = SAGEConv(256, 16)
        self.out = nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.softmax(self.out(x), dim=1)
        return x
    
def eval_node_classifier(model, graph, mask, is_test = False):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct	= (pred[mask] == graph.y[mask]).sum()
    acc = correct/mask.sum()

    true_p = ((pred[mask] == graph.y[mask]) & (pred[mask] == 1)).sum().item()
    false_p = ((pred[mask] != graph.y[mask]) & (pred[mask] == 1)).sum().item()
    false_n = ((pred[mask] != graph.y[mask]) & (pred[mask] == 0)).sum().item()

    precision = true_p / (true_p + false_p) if (true_p + false_p) > 0 else 0
    
    if is_test:
        recall = true_p / (true_p + false_n) if (true_p + false_n) > 0 else 0
        f1_score = (2*precision*recall)/(precision+recall)

        return (acc.item(), precision, recall, f1_score)
    
    return acc.item(), precision

def train_node_classifier(model, graph, optimizer, criterion, n_epochs = 200):

    train_lss_curve = []
    val_lss_curve = []  

    evaluation = {"val_accuracy":[], "val_precision":[]}

    print("\n >> Training...\n")

    print(model, "\n")

    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()

        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        train_lss_curve.append(loss.item())

        loss.backward()
        optimizer.step()

        val_loss = criterion(out[graph.val_mask], graph.y[graph.val_mask])
        val_lss_curve.append(val_loss.item())

        acc, precision = eval_node_classifier(model, graph, graph.val_mask)

        evaluation["val_accuracy"].append(acc)
        evaluation["val_precision"].append(precision)
        #evaluation["val_recall"].append(recall)

        if epoch%2 == 0:
            print (f' Epoch [{epoch:>2}/{n_epochs}] | Loss: {loss.item():.4f} | Val_loss {val_loss.item():.4f} | Val_acc {acc:.3f} - ')
            
    print()
    return model, train_lss_curve, val_lss_curve, evaluation



def do_plot(n_epochs, train_lss_curve, val_lss_curve, evaluation):
    plt.figure(figsize=(10, 2.5))  # Adjust figure size as needed

    plt.subplot(1, 2, 1)  # Subplot 1
    plt.title("\n Train_history \n")

    plt.plot(range(n_epochs), train_lss_curve, label = "Train loss")
    plt.plot(range(n_epochs), val_lss_curve, label = "Validation loss")
    plt.xlabel("\n Epochs \n")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("\n Val_Evaluation \n")
    plt.plot(range(n_epochs), evaluation["val_precision"], label = "val_precision", color = "#ABCFF0")
    plt.plot(range(n_epochs), evaluation["val_accuracy"], label = "val_accuracy", color="r")
    # plt.plot(range(n_epochs), evaluation["val_recall"], label = "val_recall", color = "#C4F0ab")
    plt.xlabel("\n Epochs \n")
    plt.grid()
    plt.legend()

    plt.subplots_adjust(wspace=0.2) 
    plt.show()



