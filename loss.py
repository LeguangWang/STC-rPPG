import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
# sys.path.append("mnt/Pytorch_rppgs/")
from log import log_warning


def loss_fn(loss_fn: str = "mse"):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """
    if loss_fn == "mse":
        return loss.MSELoss()
    elif loss_fn == "L1":
        return loss.L1Loss()
    elif loss_fn == "neg_pearson":
        return NegPearsonLoss()
    elif loss_fn == "multi_margin":
        return loss.MultiMarginLoss()
    elif loss_fn == "bce":
        return loss.BCELoss()
    elif loss_fn == "loss_huber":
        return loss.HuberLoss()
    elif loss_fn == "cosine_embedding":
        return loss.CosineEmbeddingLoss()
    elif loss_fn == "cross_entropy":
        return loss.CrossEntropyLoss()
    elif loss_fn == "ctc":
        return loss.CTCLoss()
    elif loss_fn == "bce_with_logits":
        return loss.BCEWithLogitsLoss()
    elif loss_fn == "gaussian_nll":
        return loss.GaussianNLLLoss()
    elif loss_fn == "hinge_embedding":
        return loss.HingeEmbeddingLoss()
    elif loss_fn == "KLDiv":
        return loss.KLDivLoss()
    elif loss_fn == "margin_ranking":
        return loss.MarginRankingLoss()
    elif loss_fn == "multi_label_margin":
        return loss.MultiLabelMarginLoss()
    elif loss_fn == "multi_label_soft_margin":
        return loss.MultiLabelSoftMarginLoss()
    elif loss_fn == "nll":
        return loss.NLLLoss()
    elif loss_fn == "nll2d":
        return loss.NLLLoss2d()
    elif loss_fn == "pairwise":
        return loss.PairwiseDistance()
    elif loss_fn == "poisson_nll":
        return loss.PoissonNLLLoss()
    elif loss_fn == "smooth_l1":
        return loss.SmoothL1Loss()
    elif loss_fn == "soft_margin":
        return loss.SoftMarginLoss()
    elif loss_fn == "triplet_margin":
        return loss.TripletMarginLoss()
    elif loss_fn == "triplet_margin_distance":
        return loss.TripletMarginWithDistanceLoss()
    else:
        log_warning("use implemented loss functions")
        raise NotImplementedError("implement a custom function(%s) in loss.py" % loss_fn)


def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    targets = (targets - torch.mean(targets)) / torch.std(targets)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss(predictions, targets)

# import torch
# import torch.nn as nn
#
#
# class MixedLoss(nn.Module):
#     def __init__(self, freq_weight=0.5, epsilon=1e-7):
#         super().__init__()
#         self.epsilon = epsilon
#         self.freq_weight = freq_weight  # 频域损失权重系数
#
#     def pearson_loss(self,outputs, targets):
#         '''
#         :param predictions: inference value of trained model
#         :param targets: target label of input data
#         :return: negative pearson loss
#         '''
#         rst = 0
#         # Pearson correlation can be performed on the premise of normalization of input data
#         predictions = (outputs - torch.mean(outputs)) / torch.std(outputs)
#         targets = (targets - torch.mean(targets)) / torch.std(targets)
#
#         for i in range(predictions.shape[0]):
#             sum_x = torch.sum(predictions[i])  # x
#             sum_y = torch.sum(targets[i])  # y
#             sum_xy = torch.sum(predictions[i] * targets[i])  # xy
#             sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
#             sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
#             N = predictions.shape[1]
#             pearson = (N * sum_xy - sum_x * sum_y) / (
#                 torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
#
#             rst += 1 - pearson
#
#         rst = rst / predictions.shape[0]
#         return rst
#
#     def psd_mae_loss(self, outputs, targets):
#         # 带归一化的PSD计算
#         fft_output = torch.fft.rfft(outputs, dim=-1, norm='ortho')
#         fft_target = torch.fft.rfft(targets, dim=-1, norm='ortho')
#
#         # 功率谱密度（能量归一化）
#         psd_output = torch.abs(fft_output) ** 2 / outputs.shape[-1]
#         psd_target = torch.abs(fft_target)  ** 2 / targets.shape[-1]
#
#         # 对数MAE（增强低频敏感性）
#         return torch.mean(torch.abs(torch.log(psd_output + 1e-9) - torch.log(psd_target + 1e-9)))
#
#     def forward(self, outputs, targets):
#         # 设备对齐
#         targets = targets.to(outputs.device)
#
#         # 维度扩展（兼容单样本）
#         if outputs.dim() == 1:
#             outputs = outputs.unsqueeze(0)
#             targets = targets.unsqueeze(0)
#
#         # 损失计算
#         p_loss = self.pearson_loss(outputs, targets)
#         f_loss = self.psd_mae_loss(outputs, targets)
#
#         # 加权混合
#         # return p_loss + 0.1 * f_loss
#         return 0.1 * f_loss
import torch.nn as nn
import torch.nn.functional as F
import math
#
#
class MixedLoss(nn.Module):
    def __init__(self, freq_weight=0.1, epsilon=1e-7, Fs=30):
        super().__init__()
        self.epsilon = epsilon
        self.freq_weight = freq_weight  # 频域损失权重系数
        self.Fs = Fs  # 采样率

    def pearson_loss(self, outputs, targets):
        rst = 0
        predictions = (outputs - torch.mean(outputs)) / torch.std(outputs)
        targets = (targets - torch.mean(targets)) / torch.std(targets)

        for i in range(predictions.shape[0]):
            sum_x = torch.sum(predictions[i])
            sum_y = torch.sum(targets[i])
            sum_xy = torch.sum(predictions[i] * targets[i])
            sum_x2 = torch.sum(torch.pow(predictions[i], 2))
            sum_y2 = torch.sum(torch.pow(targets[i], 2))
            N = predictions.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            rst += 1 - pearson

        return rst / predictions.shape[0]

    def psd_mae_loss(self, outputs, targets):
        """
        计算功率谱密度（PSD），并通过 Softmax 归一化后计算预测心率与真实心率的 MAE 损失
        """
        outputs = outputs.view(1, -1)
        targets = targets.view(1, -1)

        N = outputs.shape[-1]
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(outputs.device)
        unit_per_hz = self.Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        fft_targets = torch.fft.rfft(targets,dim=-1, norm='ortho')
        fft_output = torch.fft.rfft(outputs, dim=-1, norm='ortho')
        psd_targets = (torch.abs(fft_targets) ** 2) / N
        psd_output = (torch.abs(fft_output) ** 2) / N

        psd_targets = psd_targets.view(-1) / psd_output.sum()
        psd_output = psd_output.view(-1) / psd_output.sum()
        psd_targets = F.softmax(psd_targets, dim=0)
        psd_output = F.softmax(psd_output, dim=0)

        True_hr_idx = torch.argmax(psd_targets)
        predicted_hr_idx = torch.argmax(psd_output)
        True_hr = 40 + (True_hr_idx / len(psd_targets)) * (180 - 40)
        predicted_hr = 40 + (predicted_hr_idx / len(psd_output)) * (180 - 40)

        mae_loss = torch.abs(predicted_hr - True_hr)
        return mae_loss

    def forward(self, outputs, targets):
        targets = targets.to(outputs.device)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
            targets = targets.unsqueeze(0)

        p_loss = self.pearson_loss(outputs, targets)
        f_loss = self.psd_mae_loss(outputs, targets)
        # return self.freq_weight * f_loss
        return p_loss + self.freq_weight * f_loss

