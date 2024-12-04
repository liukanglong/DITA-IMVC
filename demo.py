import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import faiss
import itertools
import argparse
from meta_network import WNet, SafeNetwork, Online
from network import Network
from dataloader import load_data, MultiviewDataset, RandomSampler
from loss import Loss
from make_mask import get_mask
from evaluation import evaluate
import copy
import torch.nn.functional as F
from loss import Instance_Align_Loss, Proto_Align_Loss
from util import next_batch, get_Similarity




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset', default='NGs')
parser.add_argument("--view", type=int, default=2)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", type=int, default=128)
parser.add_argument('--lr_wnet', type=float, default=0.0004)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument("--epochs", default=120)
parser.add_argument('--lr_decay_factor', type=float, default=0.2)
parser.add_argument('--lr_decay_iter', type=int, default=20)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--initial_epochs', type=int, default=100)
parser.add_argument('--pretrain_epochs', type=int, default=100)
parser.add_argument('--train_align_epochs', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--miss_rate', default=0.1, type=float)
parser.add_argument('--T', default=5, type=int)
parser.add_argument('--iterations', default=200, type=int)
parser.add_argument('--para_loss', default=1e-3, type=float)
parser.add_argument('--a', default=1e-4, type=float)
parser.add_argument('--b', default=1e-4, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_list, Y, dims, total_view, data_size, class_num = load_data(args.dataset)
view = total_view
miss_rate = args.miss_rate
incomplete_loader = None

if args.dataset not in ['CCV']:
    for v in range(total_view):
        min_max_scaler = MinMaxScaler()
        data_list[v] = min_max_scaler.fit_transform(data_list[v])
record_data_list = copy.deepcopy(data_list)


if args.dataset == 'BDGP':
    args.initial_epochs = 30
    args.pretrain_epochs = 120
    args.iterations = 90
    args.normalized = True
    args.lmd = 0.05
    args.beta = 0.05
    args.batch = 2500

if args.dataset == 'Cora':
    args.initial_epochs = 30
    args.pretrain_epochs = 120
    args.iterations = 150
    args.normalized = True
    args.lmd = 0.05
    args.beta = 0.05
    args.batch = 2708

if args.dataset == 'MNIST_USPS':
    args.initial_epochs = 80
    args.pretrain_epochs = 100
    args.iterations = 300
    args.normalized = True
    args.lmd = 0.05
    args.beta = 0.05
    args.batch = 2000

if args.dataset == 'NGs':
    args.initial_epochs = 40
    args.pretrain_epochs = 120
    args.iterations = 100
    args.normalized = True
    args.lmd = 0.01
    args.beta = 0.01
    args.batch = 500

if args.dataset == 'synthetic3d':
    args.initial_epochs = 60
    args.pretrain_epochs = 120
    args.iterations = 100
    args.normalized = True
    args.lmd = 0.05
    args.beta = 0.05
    args.batch = 600

if args.dataset == 'CCV':
    args.initial_epochs = 30
    args.pretrain_epochs = 100
    args.iterations = 300
    args.normalized = True
    args.lmd = 0.05
    args.beta = 0.05
    args.batch = 2000

def get_model():
    return SafeNetwork(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device) # Initialize the network of the dual layer optimization module


def pretrain(com_dataset): # Pre train the complete part to obtain model parameters
    print("Initializing network parameters...")
    pretrain_model = Online(view, dims, args.feature_dim).to(device)
    loader = DataLoader(com_dataset, batch_size=args.batch_size, shuffle=True)
    opti = torch.optim.Adam(pretrain_model.params(), lr=0.0003)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.pretrain_epochs):
        for batch_idx, (xs, _, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            _,xrs = pretrain_model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v])) # reconfiguration
            loss_recon = sum(loss_list)

            criterion_ins = Instance_Align_Loss().to(device)  # alignment
            loss_list_ins = []
            zs, _ = pretrain_model.forward(xs)
            for v1 in range(view):
                v2_strat = v1 + 1
                for v2 in range(v2_strat, view):
                    z1 = zs[v1]
                    z2 = zs[v2]
                    Dx = F.cosine_similarity(z1, z2, dim=1)
                    gt = torch.ones(z1.shape[0]).to(device)
                    l_tmp2 = criterion_ins(gt, Dx)
                    loss_list_ins.append(l_tmp2)
            loss_ins_align = sum(loss_list_ins)

            loss = loss_recon + loss_ins_align * 1e-4
            # loss = loss_recon
            opti.zero_grad()
            loss.backward()
            opti.step()

    return pretrain_model.state_dict()


def bi_level_train(model, criterion, optimizer, class_num, view,
             com_loader, full_loader, mask, incomplete_ind): # Double layer optimization
    print("Updating neighbors...")
    wnet_label = WNet(class_num, 100, 1).to(device) # Weighted network
    memory = Memory()
    memory.bi = True
    wnet_label.train()
    iteration = 0

    optimizer_wnet_label = torch.optim.Adam(wnet_label.params(), lr=args.lr_wnet)

    for com_batch, incomplete_batch in zip(com_loader, incomplete_loader):
        xs, _, _ = com_batch
        incomplete_xs, _, _ = incomplete_batch
        iteration += 1
        for v in range(view):
            xs[v] = xs[v].to(device)
            incomplete_xs[v] = incomplete_xs[v].to(device)

        model.train()
        meta_net = get_model()
        meta_net.load_state_dict(model.state_dict())

        com_hs, com_qs, incomplete_hs, incomplete_qs = meta_net(xs, incomplete_xs)

        criterion_proto = Proto_Align_Loss().to(device)  # 原型对齐
        loss_list_pro = []
        qs, _ = meta_net.forward_cluster(xs)
        for v1 in range(view):
            v2_strat = v1 + 1
            for v2 in range(v2_strat, view):
                p1 = qs[v1].t()
                p2 = qs[v2].t()
                gt = torch.ones(p1.shape[0]).to(device)
                Dp = F.cosine_similarity(p1, p2, dim=1)
                l_tmp = criterion_proto(gt, Dp)
                loss_list_pro.append(l_tmp)
        loss_pro_align1 = sum(loss_list_pro)

        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w])) # Spectral contrast loss
                loss_list.append(args.lmd * criterion.forward_label1(com_qs[v], com_qs[w], args.temperature_l, args.normalized))  # Cluster allocation comparison loss
                loss_list.append(args.beta * criterion.forward_prob(com_qs[v], com_qs[w]))
        loss_hat = sum(loss_list) + loss_pro_align1 * args.a

        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v+1, view):
                l_f, l_l = (criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]),
                                args.lmd * criterion.forward_label1(incomplete_qs[v], incomplete_qs[w], args.temperature_l,args.normalized) +
                                args.beta * criterion.forward_prob(incomplete_qs[v], incomplete_qs[w]))
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)

        weight_label = wnet_label(sum(incomplete_qs)/view) # Assign weights
        norm_label = torch.sum(weight_label)

        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss_hat += (torch.sum(cost_w_features[v] * weight_label)/norm_label
                                    + torch.sum(cost_w_labels[v]*weight_label) / norm_label)
            else:
                loss_hat += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v]*weight_label)

        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_hat, meta_net.params(), create_graph=True)
        meta_net.update_params(lr_inner=args.meta_lr, source_params=grads)
        del grads # Update encoder and decoder model parameters

        com_hs, com_qs, _, _ = meta_net(xs, incomplete_xs)

        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(args.lmd * criterion.forward_label1(com_qs[v], com_qs[w], args.temperature_l,args.normalized))
                loss_list.append(args.beta * criterion.forward_prob(com_qs[v], com_qs[w]))

        l_g_meta = sum(loss_list)

        optimizer_wnet_label.zero_grad()
        l_g_meta.backward() # Optimizing weighted networks
        optimizer_wnet_label.step()

        com_hs, com_qs, incomplete_hs, incomplete_qs = model(xs, incomplete_xs)

        criterion_proto = Proto_Align_Loss().to(device)  # 原型对齐
        loss_list_pro = []
        qs, _ = model.forward_cluster(xs)
        for v1 in range(view):
            v2_strat = v1 + 1
            for v2 in range(v2_strat, view):
                p1 = qs[v1].t()
                p2 = qs[v2].t()
                gt = torch.ones(p1.shape[0]).to(device)
                Dp = F.cosine_similarity(p1, p2, dim=1)
                l_tmp = criterion_proto(gt, Dp)
                loss_list_pro.append(l_tmp)
        loss_pro_align2 = sum(loss_list_pro)

        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(args.lmd * criterion.forward_label1(com_qs[v], com_qs[w], args.temperature_l,args.normalized))
                loss_list.append(args.beta * criterion.forward_prob(com_qs[v], com_qs[w]))
        loss = sum(loss_list) + loss_pro_align2 * args.a
        # loss = sum(loss_list)

        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v + 1, view):
                l_f, l_l = (criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]),
                            args.lmd * criterion.forward_label1(incomplete_qs[v], incomplete_qs[w], args.temperature_l,args.normalized) +
                            args.beta * criterion.forward_prob(incomplete_qs[v], incomplete_qs[w]))
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)

        with torch.no_grad():
            weight_label = wnet_label(sum(incomplete_qs) / view)
            norm_label = torch.sum(weight_label)

        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss += (torch.sum(cost_w_labels[v] * weight_label) / norm_label
                         + torch.sum(cost_w_features[v] * weight_label) / norm_label)
            else:
                loss += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v] * weight_label)

        optimizer.zero_grad()
        loss.backward() # Optimize encoder and decoder networks
        optimizer.step()

        memory.update_feature(model, full_loader, mask, incomplete_ind, iteration) # Updating feature representations and repairing missing views

    acc, nmi, pur = valid(model, mask)

    return acc, nmi, pur


def valid(model, mask):
    pred_vec = []
    with torch.no_grad():
        input_data = []
        for v in range(view):
            data_v = torch.from_numpy(record_data_list[v]).to(device)
            input_data.append(data_v)
        output, _ = model.forward_cluster(input_data)
        for v in range(view):
            miss_ind = mask[:, v] == 0
            output[v][miss_ind] = 0
        sum_ind = np.sum(mask, axis=1, keepdims=True)
        output = sum(output)/torch.from_numpy(sum_ind).to(device)
        pred_vec.extend(output.detach().cpu().numpy())

    pred_vec = np.argmax(np.array(pred_vec), axis=1)
    acc, nmi, pur = evaluate(Y, pred_vec)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(acc, nmi, pur))
    return acc, nmi, pur


class Memory:
    def __init__(self):
        self.features = None
        self.alpha = args.alpha
        self.interval = args.interval
        self.bi = False

    def update_feature(self, model, loader, mask, incomplete_ind, epoch):
        model.eval()
        indices = []
        if epoch == 1:
            features = []
            for v in range(view):
                features.append([])

            for _, (xs, y, _) in enumerate(loader): # Obtain feature representation
                for v in range(view):
                    xs[v] = xs[v].to(device)
                with torch.no_grad():
                    if self.bi:
                        hs, _, _ = model.forward_xs(xs)
                    else:
                        hs, _, _ = model(xs)
                    for v in range(view):
                        fea = hs[v].detach().cpu().numpy()
                        features[v].extend(fea)
            for v in range(view):
                features[v] = np.array(features[v])

            self.features = features
            final_batch = args.batch
            full_dataset2 = MultiviewDataset(view, self.features, Y)
            full_loader2 = DataLoader(full_dataset2, batch_size=final_batch, shuffle=False, drop_last=False)
            for batch_idx, (xs, y, _) in enumerate(full_loader2):
                for v in range(view):
                    xs[v] = xs[v].clone().detach()
                    xs[v] = torch.squeeze(xs[v]).to(device)
                cossim_mat = []
                for v in range(view):
                    sim_mat = get_Similarity(xs[v], xs[v])  # Calculate sample similarity matrix
                    diag = torch.diag(sim_mat)
                    sim_diag = torch.diag_embed(diag)
                    sim_mat = sim_mat - sim_diag
                    cossim_mat.append(sim_mat)

                for i in range(xs[0].shape[0]):
                    for v in range(view):
                        if mask[final_batch * batch_idx + i, v] == 0:
                            predicts = []
                            n_w = 0
                            for x in range(data_size):
                                # only the available views are selected as neighbors
                                for w in range(view):
                                    if w != v and mask[final_batch * batch_idx + i, w] != 0:
                                        vec_tmp = cossim_mat[w][i]
                                        _, indices = torch.sort(vec_tmp, descending=True)
                                        # for n_w in range(neigh_w.shape[0]):
                                        if mask[indices[n_w], v] != 0 and mask[indices[n_w], w] != 0:
                                            predicts.append(data_list[v][indices[n_w]])
                                if len(predicts) != 0:
                                    break
                                else:
                                    n_w = n_w + 1

                            fill_sample = np.mean(predicts, axis=0)
                            data_list[v][final_batch * batch_idx + i] = fill_sample

            return data_list

        elif epoch % self.interval == 0: # Determine if it is the first repair
            features = []
            for v in range(view):
                features.append([])

            for _, (xs, y, _) in enumerate(loader):
                for v in range(view):
                    xs[v] = xs[v].to(device)
                with torch.no_grad():
                    if self.bi:
                        hs, _, _ = model.forward_xs(xs)
                    else:
                        hs, _, _ = model(xs)
                    for v in range(view):
                        fea = hs[v].detach().cpu().numpy()
                        features[v].extend(fea)
            for v in range(view):
                features[v] = np.array(features[v])

            cur_features = features
            for v in range(view):
                f_v = (1-self.alpha)*self.features[v] + self.alpha*cur_features[v]
                self.features[v] = f_v/np.linalg.norm(f_v, axis=1, keepdims=True)
            final_batch = args.batch
            full_dataset2 = MultiviewDataset(view, self.features, Y)
            full_loader2 = DataLoader(full_dataset2, batch_size=final_batch, shuffle=False, drop_last=False)
            for batch_idx, (xs, y, _) in enumerate(full_loader2):
                for v in range(view):
                    xs[v] = xs[v].clone().detach()
                    xs[v] = torch.squeeze(xs[v]).to(device)
                cossim_mat = []
                for v in range(view):
                    sim_mat = get_Similarity(xs[v], xs[v])  # Calculate sample similarity matrix
                    diag = torch.diag(sim_mat)
                    sim_diag = torch.diag_embed(diag)
                    sim_mat = sim_mat - sim_diag
                    cossim_mat.append(sim_mat)

                for i in range(xs[0].shape[0]):
                    for v in range(view):
                        if mask[final_batch * batch_idx + i, v] == 0:
                            predicts = []
                            n_w = 0
                            for x in range(data_size):
                                # only the available views are selected as neighbors
                                for w in range(view):
                                    if w != v and mask[final_batch * batch_idx + i, w] != 0:
                                        vec_tmp = cossim_mat[w][i]
                                        _, indices = torch.sort(vec_tmp, descending=True)
                                        # for n_w in range(neigh_w.shape[0]):
                                        if mask[indices[n_w], v] != 0 and mask[indices[n_w], w] != 0:
                                            predicts.append(data_list[v][indices[n_w]])
                                if len(predicts) != 0:
                                    break
                                else:
                                    n_w = n_w + 1

                            fill_sample = np.mean(predicts, axis=0)
                            data_list[v][final_batch * batch_idx + i] = fill_sample
            if self.bi:
                make_imputation(data_list, incomplete_ind)
            return data_list


def make_imputation(data_list, incomplete_ind):
    global incomplete_loader
    incomplete_data = []
    for v in range(view): # Obtain repaired attempt data
        incomplete_data.append(data_list[v][incomplete_ind])
    incomplete_label = Y[incomplete_ind]
    incomplete_dataset = MultiviewDataset(view, incomplete_data, incomplete_label)
    incomplete_loader = DataLoader(
        incomplete_dataset, args.batch_size, drop_last=True,
        sampler=RandomSampler(len(incomplete_dataset), args.iterations * args.batch_size)
    )


def initial(com_dataset, full_loader, criterion, mask, incomplete_ind):
    print("Initializing neighbors...") # 初始化邻居
    online_net = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)
    loader = DataLoader(com_dataset, batch_size=256, shuffle=True, drop_last=True)
    mse_loader = DataLoader(com_dataset, batch_size=256, shuffle=True)
    opti = torch.optim.Adam(online_net.parameters(), lr=0.0003, weight_decay=0.) # 优化器
    mse = torch.nn.MSELoss()

    memory = Memory()
    memory.interval = 1
    epochs = args.initial_epochs # 30

    # pretraining on complete data
    for e in range(1, 201):
        for xs, _, _ in mse_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)

            xrs = online_net.forward_mse(xs)

            loss_list = []
            for v in range(view):
                loss_list.append(mse(xrs[v], xs[v]))
            loss_recon = sum(loss_list)

            criterion_ins = Instance_Align_Loss().to(device)
            loss_list_ins = []
            zs = online_net.forward_ms(xs)
            for v1 in range(view):
                v2_strat = v1 + 1
                for v2 in range(v2_strat, view):
                    z1 = zs[v1]
                    z2 = zs[v2]
                    Dx = F.cosine_similarity(z1, z2, dim=1)
                    gt = torch.ones(z1.shape[0]).to(device)
                    l_tmp2 = criterion_ins(gt, Dx)
                    loss_list_ins.append(l_tmp2)
            loss_ins_align = sum(loss_list_ins)

            criterion_proto = Proto_Align_Loss().to(device)   #原型对齐
            loss_list_pro = []
            qs, _ = online_net.forward_cluster(xs)
            for v1 in range(view):
                v2_strat = v1 + 1
                for v2 in range(v2_strat, view):
                    p1 = qs[v1].t()
                    p2 = qs[v2].t()
                    gt = torch.ones(p1.shape[0]).to(device)
                    Dp = F.cosine_similarity(p1, p2, dim=1)
                    l_tmp = criterion_proto(gt, Dp)
                    loss_list_pro.append(l_tmp)
            loss_pro_align = sum(loss_list_pro)

            loss = loss_recon + loss_pro_align * args.a + loss_ins_align * args.b
            # loss = loss_recon
            opti.zero_grad()
            loss.backward()
            opti.step()


    for e in range(1, epochs+1):
        for xs, _, _ in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)

            hs, qs, _ = online_net(xs)

            loss_list = []
            for v in range(view):
                for w in range(v+1, view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w])) # 光谱对比损失
                    loss_list.append(args.lmd*criterion.forward_label1(qs[v], qs[w], args.temperature_l, args.normalized)) # 聚类分配损失
                    loss_list.append(args.beta*criterion.forward_prob(qs[v], qs[w]))
            loss = sum(loss_list)

            opti.zero_grad()
            loss.backward()
            opti.step()

    # initial neighbors by the pretrain model
    data_list = memory.update_feature(online_net, full_loader, mask, incomplete_ind, epoch=1)
    make_imputation(data_list, incomplete_ind) # 数据填充


def main():
    result_record = {"ACC": [], "NMI": [], "PUR": []}
    for t in range(1, args.T+1):
        print("--------Iter:{}--------".format(t))
        X_list = copy.deepcopy(record_data_list)
        mask = get_mask(view, data_size, miss_rate)
        sum_vec = np.sum(mask, axis=1, keepdims=True) # Obtain missing data through mask matrix
        complete_index = (sum_vec[:, 0]) == view
        mv_data = []
        for v in range(view):
            mv_data.append(X_list[v][complete_index])
        mv_label = Y[complete_index]
        com_dataset = MultiviewDataset(view, mv_data, mv_label) # Obtain complete partial data
        com_loader = DataLoader(
            com_dataset, 256, drop_last=True,
            sampler=RandomSampler(len(com_dataset), args.iterations * args.batch_size)
        )
        full_dataset = MultiviewDataset(view, X_list, Y) # Original complete data
        full_loader = DataLoader(full_dataset, args.batch_size, shuffle=False, drop_last=False)
        incomplete_ind = (sum_vec[:, 0]) != view # Incomplete data index

        model = get_model()
        state_dict = pretrain(com_dataset) # Obtain pre trained model parameters
        model.load_state_dict(state_dict, strict=False)
        optimizer = torch.optim.Adam(model.params(), lr=0.0003, weight_decay=0.) # optimizer
        criterion = Loss(args.batch_size, class_num, view, device) # Comparative loss
        initial(com_dataset, full_loader, criterion, mask, incomplete_ind) # Missing data repair
        acc, nmi, pur = bi_level_train(model, criterion, optimizer, class_num, view, com_loader,
                 full_loader, mask, incomplete_ind) # Cluster evaluation
        result_record["ACC"].append(acc)
        result_record["NMI"].append(nmi)
        result_record["PUR"].append(pur)

    print("----------------Training Finish----------------")
    print("----------------Final Results----------------")
    print("ACC (mean) = {:.4f} ACC (std) = {:.4f}".format(np.mean(result_record["ACC"]), np.std(result_record["ACC"])))
    print("NMI (mean) = {:.4f} NMI (std) = {:.4f}".format(np.mean(result_record["NMI"]), np.std(result_record["NMI"])))
    print("PUR (mean) = {:.4f} PUR (std) = {:.4f}".format(np.mean(result_record["PUR"]), np.std(result_record["PUR"])))


if __name__ == '__main__':
    main()
