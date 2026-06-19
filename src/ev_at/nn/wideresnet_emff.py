### This code is extracted from https://github.com/zissermannn/emff/tree/main

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(
        label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True
    )

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return A + B


def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMC(nn.Module):
    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super().__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList(
            [Classifier(classifier_dims[i], self.classes) for i in range(self.views)]
        )

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(
                b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes)
            )
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / (
                (1 - C).view(-1, 1).expand(b[0].shape)
            )
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, X, y, global_step, is_eval=False):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            if not is_eval:
                loss += ce_loss(
                    y, alpha[v_num], self.classes, global_step, self.lambda_epochs
                )
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        if not is_eval:
            loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
            loss = torch.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super().__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i + 1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers - 1], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h


class BasicBlockEMFF(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlockEMFF(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetEMFF(nn.Module):
    def __init__(
        self,
        depth=34,
        num_classes=10,
        widen_factor=10,
        dropRate=0.0,
        tau=0.1,
        image_size=(32, 32),
        dims=[[160], [320], [640]],
        epoch=200,
    ):
        super().__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlockEMFF
        self.image_size = image_size
        self.tau = tau
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.block1 = NetworkBlockEMFF(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        self.sub_block1 = NetworkBlockEMFF(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        self.block2 = NetworkBlockEMFF(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        self.block3 = NetworkBlockEMFF(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        # self.separation = Separation(size=(640, int(self.image_size[0] / 4), int(self.image_size[1] / 4)), tau=self.tau)
        # self.recalibration = Recalibration(size=(640, int(self.image_size[0] / 4), int(self.image_size[1] / 4)))
        # self.aux = nn.Sequential(nn.Linear(640, num_classes))
        self.dim = dims
        self.views = len(self.dim)
        self.tmc = TMC(num_classes, self.views, self.dim, epoch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, labels, epoch, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []
        tmc_f = []

        out = self.conv1(x)
        out = self.block1(out)
        tmc_f.append(torch.nn.AdaptiveAvgPool2d(1)(out).reshape(out.shape[0], -1))
        out = self.block2(out)
        tmc_f.append(torch.nn.AdaptiveAvgPool2d(1)(out).reshape(out.shape[0], -1))
        out = self.block3(out)
        tmc_f.append(torch.nn.AdaptiveAvgPool2d(1)(out).reshape(out.shape[0], -1))
        out = self.relu(self.bn1(out))

        # r_feat, nr_feat, mask = self.separation(out, is_eval=is_eval)
        # r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        # r_outputs.append(r_out)
        # nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        # nr_outputs.append(nr_out)

        # rec_feat = self.recalibration(nr_feat, mask)
        # rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        # rec_outputs.append(rec_out)

        # out = r_feat + rec_feat

        # out = F.avg_pool2d(out, 8)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        tmc_f.append(out)
        # Here we included is_eval in TMC to avoid tmc_loss in evaluation mode
        evidences, evidence_b, tmc_loss = self.tmc(
            tmc_f, labels, epoch, is_eval=is_eval
        )

        return out, r_outputs, nr_outputs, rec_outputs, evidence_b, tmc_loss


class WideResNetEMFF3410(WideResNetEMFF):
    def __init__(self, num_classes=10):
        super().__init__(
            depth=34,
            num_classes=num_classes,
            widen_factor=10,
            tau=0.1,
            image_size=(32, 32),
            dims=[[160], [320], [640], [num_classes]],
            epoch=200,
        )
