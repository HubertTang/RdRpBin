from motif_gcn_scripts.data import load_data, preprocess_features, preprocess_adj, sample_mask
from motif_gcn_scripts.utils import masked_loss, masked_acc
from motif_gcn_scripts import model
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle as pkl
import  scipy.sparse as sp
from scipy.special import softmax
import torch
from torch import optim



def train_motif_GCN(args, motif_gcn_temp_dir, train_csv_path, pred_out_path, force_cpu=False):
    """Train the motif GCN model.
    """
    
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        print("Running with cpu")

    adj        = pkl.load(open(f"{motif_gcn_temp_dir}/read.graph",'rb'))
    labels     = pkl.load(open(f"{motif_gcn_temp_dir}/read.label",'rb'))
    features   = pkl.load(open(f"{motif_gcn_temp_dir}/read.feature",'rb'))
    test_to_id = pkl.load(open(f"{motif_gcn_temp_dir}/read.dict",'rb'))
    train_idx  = pkl.load(open(f"{motif_gcn_temp_dir}/read.train_idx",'rb'))
    valid_idx  = pkl.load(open(f"{motif_gcn_temp_dir}/read.valid_idx",'rb'))
    test_idx   = pkl.load(open(f"{motif_gcn_temp_dir}/read.test_idx",'rb'))

    # load labels
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    test_idx = np.array(test_idx)
    labels = np.array(labels)

    y_train = np.zeros(labels.shape)
    y_valid = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    train_mask = sample_mask(train_idx, labels.shape[0])
    valid_mask = sample_mask(valid_idx, labels.shape[0])
    test_mask = sample_mask(test_idx, labels.shape[0])

    y_train[train_mask] = labels[train_mask]
    y_valid[valid_mask] = labels[valid_mask]
    y_test[test_mask] = labels[test_mask]

    # load features
    features = sp.csc_matrix(features)

    print('adj:', adj.shape)
    print('features:', features.shape)
    print('y:', y_train.shape, y_valid.shape, y_test.shape) # y_val.shape, 
    print('mask:', train_mask.shape, valid_mask.shape, test_mask.shape) # val_mask.shape

    features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
    supports = preprocess_adj(adj)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        num_classes = max(labels)+1
        train_label = torch.from_numpy(y_train).long().to(device)
        valid_label = torch.from_numpy(y_valid).long().to(device)
        test_label = torch.from_numpy(y_test).long().to(device)
        train_mask = torch.from_numpy(train_mask.astype(np.bool)).to(device)
        valid_mask = torch.from_numpy(valid_mask.astype(np.bool)).to(device)
        test_mask = torch.from_numpy(test_mask.astype(np.bool)).to(device)

        i = torch.from_numpy(features[0]).long().to(device)
        v = torch.from_numpy(features[1]).to(device)
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

        i = torch.from_numpy(supports[0]).long().to(device)
        v = torch.from_numpy(supports[1]).to(device)
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    else:
        train_label = torch.from_numpy(y_train).long()
        valid_label = torch.from_numpy(y_valid).long()
        num_classes = max(labels)+1
        train_mask = torch.from_numpy(train_mask.astype(np.bool))
        test_label = torch.from_numpy(y_test).long()
        valid_mask = torch.from_numpy(valid_mask.astype(np.bool))
        test_mask = torch.from_numpy(test_mask.astype(np.bool))

        i = torch.from_numpy(features[0]).long()
        v = torch.from_numpy(features[1])
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float()

        i = torch.from_numpy(supports[0]).long()
        v = torch.from_numpy(supports[1])
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float()

    print('x :', feature)
    print('sp:', support)
    num_features_nonzero = feature._nnz()
    feat_dim = feature.shape[1]

    # calculate the loss weights
    all_data = pd.read_csv(train_csv_path, header=None)[0].values
    all_label = [i for i in range(num_classes)]
    num_each_class = np.array([np.count_nonzero(all_data == l) for l in all_label])
    class_weights = num_each_class.max()/ num_each_class

    def accuracy(out, mask):
        pred = np.argmax(out, axis = 1)
        mask_pred = np.array([pred[i] for i in range(len(labels)) if mask[i] == True])
        mask_label = np.array([labels[i] for i in range(len(labels)) if mask[i] == True])
        return np.sum(mask_label == mask_pred)/len(mask_pred)

    net = model.GCN(feat_dim, num_classes, num_features_nonzero, args.hidden)

    def evaluate():
        net.eval()
        with torch.no_grad():
            eval_out = net((feature, support))
            eval_loss = masked_loss(eval_out, valid_label, valid_mask, class_weights)
            eval_acc = accuracy(eval_out.detach().cpu().numpy(), valid_mask.detach().cpu().numpy())
        
        return eval_loss, eval_acc

    # load the model trained from stage one
    # if args.trained_model == "None":
    #     pass
    # else:
        # net.load_state_dict(torch.load(args.trained_model))

    if torch.cuda.is_available():
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)#args.learning_rate

    val_losses = []
    # _ = net.train()
    for epoch in range(args.epochs):
        _ = net.train()
        # forward pass
        out = net((feature, support))
        #out = out[0]
        loss = masked_loss(out, train_label, train_mask, class_weights)
        loss += args.weight_decay * net.l2_loss()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculating the acc
        _ = net.eval()
        out = net((feature, support))
        if torch.cuda.is_available():
            acc_train = accuracy(out.detach().cpu().numpy(), train_mask.detach().cpu().numpy())
        else:
            acc_train = accuracy(out.detach().numpy(), train_mask.detach().numpy())
        #acc_test = accuracy(out.detach().cpu().numpy(), test_mask.detach().cpu().numpy())
        # evaluation
        val_loss, val_acc = evaluate()
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, train_loss={loss.item()}, train_acc={acc_train}, val_loss={val_loss}, val_acc={val_acc}")
        
    torch.save(net.state_dict(), f"{motif_gcn_temp_dir}/motif_GCN_model.pt")

    net.eval()
    out = net((feature, support))
    if torch.cuda.is_available():
        out = out.cpu().detach().numpy()
    else:
        out = out.detach().numpy()
    out = softmax(out, axis=1)
    pred = np.argmax(out, axis = 1)

    mode = "testing"

    if num_classes == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    
    elif num_classes == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}
    
    # pred_out_path = os.path.join(work_dir, 'log/motif_GCN/prediction.csv')
    # os.makedirs(os.path.join(work_dir, 'log/motif_GCN'))
    with open(pred_out_path, 'w') as f_out:
        # _ = f_out.write("contig_names,prediction\n")
        for key in test_to_id.keys():
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "," + str(np.max(out[test_to_id[key]])) + "\n")
            # if labels[test_to_id[key]] == -1:
            #     _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "\n")
            # else:
                # _ = f_out.write(str(key) + "," + str(pred_to_label[labels[test_to_id[key]]]) + "\n")


