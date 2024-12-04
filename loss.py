import torch
import sys
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, view, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device
        self.view = view

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_samples2(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m1, m2], dim=1)
        mask2 = torch.cat([m2, m1], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask

    def mask_correlated_samples3(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m2, m1], dim=1)
        mask2 = torch.cat([m1, m2], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask

    def forward_feature(self, z1, z2, r=3.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1] # 计算损失的第一部分。首先计算z1和z2的点积，然后乘以z1的维度，再取负数，并求均值。
        square_term = torch.matmul(z1, z2.T) ** 2 # 计算z1和z2的内积，并将结果的每个元素平方。
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1) #光谱对比损失

        return loss_part1 + loss_part2

    def forward_feature2(self, z1, z2, r=3.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.sum(z1*z2, dim=1, keepdim=True)/z1.shape[0]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.sum(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1), dim=1,
                               keepdim=True) \
                     / (z1.shape[0] * (z1.shape[0] - 1))

        return loss_part1 + loss_part2

    # def forward_label(self, q_i, q_j):
    #     p_i = q_i.sum(0).view(-1)
    #     p_i /= p_i.sum()
    #     ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    #     p_j = q_j.sum(0).view(-1)
    #     p_j /= p_j.sum()
    #     ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    #     entropy = ne_i + ne_j
    #
    #     q_i = q_i.t()
    #     q_j = q_j.t()
    #     N = 2 * self.class_num
    #
    #     q = torch.cat((q_i, q_j), dim=0)
    #
    #     sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0))
    #     sim_i_j = torch.diag(sim, self.class_num)
    #     sim_j_i = torch.diag(sim, -self.class_num)
    #
    #     positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     mask = self.mask_correlated_samples2(N)
    #     negative_clusters = sim[mask].reshape(N, -1)
    #
    #     labels = torch.zeros(N).to(positive_clusters.device).long()
    #     logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    #     loss = self.criterion(logits, labels)
    #     loss /= N
    #
    #     return loss + entropy  # 聚类概率损失


    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = (p_j * torch.log(p_j)).sum()

        entropy = ne_i + ne_j

        return entropy

    def forward_label1(self, q_i, q_j, temperature_l, normalized=False):

        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0) # 将 q_i 和 q_j 进行转置，并将它们在维度 0 上进行拼接

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)# 公式 （5）

        sim_i_j = torch.diag(sim, self.class_num) # 正样本
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N) # 生成掩码矩阵
        negative_clusters = sim[mask].reshape(N, -1) # 负样本

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t() #公式（4）


class Instance_Align_Loss(nn.Module):
    def __init__(self):
        super(Instance_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2

class Proto_Align_Loss(nn.Module):
    def __init__(self):
        super(Proto_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss1 = mse(gt, P)

        return Loss1