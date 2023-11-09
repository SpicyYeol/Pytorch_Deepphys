import os
import math
import scipy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import torch.utils.checkpoint as cp
from torch.autograd import Variable
import torch.fft as fft

from log import log_warning
from utils.funcs import _nearest_power_of_2, normalize_torch


def loss_fn(loss_name):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """

    if loss_name == "MSE":
        return loss.MSELoss()
    elif loss_name == "fft":
        return fftLoss()
    elif loss_name == "LSTCrPPG":
        return LSTCrPPGLoss()
    elif loss_name == "L1":
        return loss.L1Loss()
    elif loss_name == "neg_pearson":
        return NegPearsonLoss()
    elif loss_name == "multi_margin":
        return loss.MultiMarginLoss()
    elif loss_name == "bce":
        return loss.BCELoss()
    elif loss_name == "huber":
        return loss.HuberLoss()
    elif loss_name == "cosine_embedding":
        return loss.CosineEmbeddingLoss()
    elif loss_name == "cross_entropy":
        return loss.CrossEntropyLoss()
    elif loss_name == "ctc":
        return loss.CTCLoss()
    elif loss_name == "bce_with_logits":
        return loss.BCEWithLogitsLoss()
    elif loss_name == "gaussian_nll":
        return loss.GaussianNLLLoss()
    elif loss_name == "hinge_embedding":
        return loss.HingeEmbeddingLoss()
    elif loss_name == "KLDiv":
        return loss.KLDivLoss()
    elif loss_name == "margin_ranking":
        return loss.MarginRankingLoss()
    elif loss_name == "multi_label_margin":
        return loss.MultiLabelMarginLoss()
    elif loss_name == "multi_label_soft_margin":
        return loss.MultiLabelSoftMarginLoss()
    elif loss_name == "nll":
        return loss.NLLLoss()
    elif loss_name == "nll2d":
        return loss.NLLLoss2d()
    elif loss_name == "pairwise":
        return loss.PairwiseDistance()
    elif loss_name == "poisson_nll":
        return loss.PoissonNLLLoss()
    elif loss_name == "smooth_l1":
        return loss.SmoothL1Loss()
    elif loss_name == "soft_margin":
        return loss.SoftMarginLoss()
    elif loss_name == "triplet_margin":
        return loss.TripletMarginLoss()
    elif loss_name == "triplet_margin_distance":
        return loss.TripletMarginWithDistanceLoss()
    elif loss_name == "RhythmNetLoss":
        return RhythmNetLoss()
    elif loss_name == "BVPVelocityLoss":
        return BVPVelocityLoss()
    elif loss_name == "CLGDLoss":
        return CurriculumLearningGuidedDynamicLoss()
    elif loss_name == "PDLoss":
        return PeakDetectionLoss()
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
    targets = targets[:, :]
    # predictions = torch.squeeze(predictions)
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions, dim=-1, keepdim=True)) / torch.std(predictions, dim=-1,
                                                                                            keepdim=True)
    targets = (targets - torch.mean(targets, dim=-1, keepdim=True)) / torch.std(targets, dim=-1, keepdim=True)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1] if len(predictions.shape) > 1 else 1
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst


def peak_mse(predictions, targets):
    rst = 0
    targets = targets[:, :]


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)
        if len(targets.shape) == 1:
            targets = targets.view(1, -1)
        return neg_Pearson_Loss(predictions, targets)


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, predictions, targets):
        neg = neg_Pearson_Loss(predictions, targets)
        loss_func = nn.L1Loss()
        predictions = torch.fft.fft(predictions, dim=1, norm="forward")
        targets = torch.fft.fft(targets, dim=1, norm="forward")
        loss = loss_func(predictions, targets)
        return loss + neg


class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = RhythmNet_autograd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, resnet_outputs, gru_outputs, target):
        frame_rate = 25.0
        # resnet_outputs, gru_outputs, _ = outputs
        # target_array = target.repeat(1, resnet_outputs.shape[1])
        l1_loss = self.l1_loss(resnet_outputs, target)
        smooth_loss_component = self.smooth_loss(gru_outputs)

        loss = l1_loss + self.lambd * smooth_loss_component
        return loss

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        # hr_mean = self.gru_outputs_considered.mean()
        for hr_t in self.gru_outputs_considered:
            # custom_fn = RhythmNet_autograd.apply
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]


class RhythmNet_autograd(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output = torch.zeros(1).to('cuda')

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t

        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1 / ctx.T) * torch.sign(ctx.hr_mean - hr)

        output = (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None


def stft(input_signal):
    stft_sig = torch.stft(input_signal, n_fft=1024, hop_length=512, win_length=1024, window=torch.hamming_window(1024),
                          center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    return stft_sig


def phase_diff_loss(pred, gt):
    pred_phase = torch.angle(pred)
    gt_phase = torch.angle(gt)
    loss = torch.abs(torch.sum(torch.exp(1j * (pred_phase - gt_phase)))) / pred.size(0)
    return loss


def phase_correlation_loss(input, target):
    # Define the forward pass for computing the phase correlation matrix
    def forward(x):
        # Compute the STFTs of the input and target signals
        input_stft = torch.stft(x, n_fft=input.shape[1], window=torch.hann_window(input.shape[1], device=input.device),
                                center=False)
        target_stft = torch.stft(target, n_fft=target.shape[1],
                                 window=torch.hann_window(target.shape[1], device=target.device), center=False)

        # Compute the complex conjugate of the target STFT
        target_conj = torch.conj(target_stft)

        # Compute the phase correlation matrix
        corr_matrix = input_stft * target_conj
        corr_matrix /= torch.abs(corr_matrix)
        corr_matrix = torch.fft.irfft(corr_matrix, dim=1)

        return corr_matrix

    # Compute the phase correlation matrix using memory checkpointing
    corr_matrix = cp.checkpoint(forward, input)

    # Compute the index of the maximum correlation value for each batch element
    max_corr_idx = torch.argmax(corr_matrix, dim=1)

    # Compute the phase correlation coefficient loss for the batch
    loss = 1.0 - torch.mean(torch.cos(torch.tensor(2.0 * np.pi * max_corr_idx / input.shape[1], device=input.device)))

    return loss


def mutual_information_loss(signal1, signal2, num_bins=32):
    # Compute the joint histogram
    hist2d = torch.histc(torch.stack([signal1, signal2], dim=1), bins=num_bins)

    # Compute the marginal histograms
    hist1 = torch.histc(signal1, bins=num_bins)
    hist2 = torch.histc(signal2, bins=num_bins)

    eps = 1e-8
    hist2d = hist2d + eps
    hist1 = hist1 + eps
    hist2 = hist2 + eps

    # Compute the probabilities and entropies
    p12 = hist2d / torch.sum(hist2d)
    p1 = hist1 / torch.sum(hist1)
    p2 = hist2 / torch.sum(hist2)
    H1 = -torch.sum(p1 * torch.log2(p1))
    H2 = -torch.sum(p2 * torch.log2(p2))

    # Compute the mutual information
    MI = torch.sum(p12 * torch.log2(p12 / (torch.outer(p1, p2))))

    # Normalize the mutual information
    NMI = MI / (0.5 * (H1 + H2))

    # Return the negative mutual information as the loss
    return NMI / signal1.shape[0]


def peak_loss(y_true, y_pred, alpha=0.5, beta=1.0):
    def find_peaks_torch(signal, height=None, distance=None):
        signal_np = signal.detach().cpu().numpy()
        peaks, _ = find_peaks(signal_np, height=height, distance=distance)
        return torch.tensor(peaks, dtype=torch.int64)

    def find_peaks_negative(signal, height=None, distance=None):
        signal_np = -1.0 * signal.detach().cpu().numpy()
        peaks, _ = find_peaks(signal_np, height=height, distance=distance)
        return torch.tensor(peaks, dtype=torch.int64)

    def find_peak_freq_torch(signal, fs=30):
        signal_np = signal.detach().cpu().numpy()
        N = _nearest_power_of_2(signal_np.shape[0])
        f_ppg, pxx_ppg = scipy.signal.periodogram(signal_np, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= 0.75) & (f_ppg <= 2.5))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        peak_freq = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0]
        return peak_freq

    def find_peaks_values(signal, peaks):
        return signal[peaks]

    batch_size = y_true.size(0)
    total_loss = 0

    with torch.no_grad():
        for i in range(batch_size):
            # Find the peaks in the true and predicted signals
            y_true_peaks = find_peaks_torch(y_true[i])
            y_pred_peaks = find_peaks_torch(y_pred[i])

            y_pred_peak_values = find_peaks_values(y_pred[i], y_pred_peaks)

            # Calculate the difference in the number of peaks
            peak_count_difference = np.abs(y_true_peaks.size(0) - y_pred_peaks.size(0))
            peak_value_difference = torch.abs(1 - y_pred_peak_values.mean())

            y_true_peaks = find_peaks_negative(y_true[i])
            y_pred_peaks = find_peaks_negative(y_pred[i])

            y_pred_peak_values = find_peaks_values(y_pred[i], y_pred_peaks)

            # Calculate the difference in the number of peaks
            neg_peak_count_difference = np.abs(y_true_peaks.size(0) - y_pred_peaks.size(0))

            neg_peak_value_difference = torch.abs(1 - y_pred_peak_values.mean())

            y_true_peak_freq = find_peak_freq_torch(y_true[i])
            y_pred_peak_freq = find_peak_freq_torch(y_pred[i])

            # Calculate the difference in peak frequency
            freq_diff = torch.abs(torch.tensor(y_true_peak_freq - y_pred_peak_freq))

            # Combine the losses
            loss = alpha * (
                    peak_count_difference + neg_peak_count_difference + peak_value_difference + neg_peak_value_difference) + freq_diff  # + beta * peak_position_difference
            total_loss += loss

    return total_loss / batch_size


class BVPVelocityLoss(nn.Module):
    def __init__(self):
        super(BVPVelocityLoss, self).__init__()
        self.trip = nn.TripletMarginLoss()
        # a / pos / neg

    def forward(self, predictions, targets):
        # [f,l,r,t]
        # (f >-< t,f <->r) (f >-< t, f<->l)
        # (l >-< t, l <->f) (l >-<r, l <-> f)
        # (r >-< t, r <->f) (r >-<r, r <-> f)

        pearson = neg_Pearson_Loss(predictions, targets)
        # NMI = mutual_information_loss(predictions, targets)
        # phase = phase_correlation_loss(predictions, targets)

        # perd_loss = periodic_signal_loss(targets,predictions)

        loss = pearson + peak_loss(targets, predictions) + derivative_loss(predictions, targets)  # + NMI + phase
        return loss


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


# derivative loss for bvp
def derivative_loss(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    batch = predictions.shape[0]

    loss = 0

    for i in range(len(predictions)):
        # 1st derivative
        predictions[i] = np.gradient(predictions[i])
        targets[i] = np.gradient(targets[i])
        loss += cos_sim(predictions[i], targets[i])
        # 2nd derivative
        predictions[i] = np.gradient(predictions[i])
        targets[i] = np.gradient(targets[i])
        loss += cos_sim(predictions[i], targets[i])

    return 2 - loss / batch


def periodic_signal_loss(signal, pred_period):
    for s in signal:
        periods = s.unfold(1, pred_period, pred_period).squeeze(0)
        min_values, _ = torch.min(periods, dim=1)
        max_values, _ = torch.max(periods, dim=1)

    # 예측된 주기를 정수로 변환
    pred_period = int(pred_period.item())

    # 주기의 시작 인덱스를 찾기 위해 신호를 여러 개의 주기로 분할
    periods = signal.unfold(1, pred_period, pred_period).squeeze(0)

    # 각 주기의 최소값과 최대값 찾기
    min_values, _ = torch.min(periods, dim=1)
    max_values, _ = torch.max(periods, dim=1)

    # 최소값과 최대값의 차이 계산
    min_max_diff = torch.mean(max_values - min_values)

    return min_max_diff


def autocorrelation(signal, max_lag=None):
    if max_lag is None:
        max_lag = signal.shape[-1] // 2

    acf = torch.zeros(max_lag)
    signal_mean = torch.mean(signal)

    for lag in range(max_lag):
        acf[lag] = torch.mean((signal[:, :-lag - 1] - signal_mean) * (signal[:, lag:] - signal_mean))

    return acf


def bandpass_filter(data, lowcut=0.8, highcut=2.5, fs=30, order=5):
    # Design bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    # Apply bandpass filter
    y = sosfiltfilt(sos, data)
    return y


class LSTCrPPGLoss(nn.Module):
    def __init__(self):
        super(LSTCrPPGLoss, self).__init__()
        self.timeLoss = nn.MSELoss()
        self.lambda_value = 0.2
        self.alpha = 1.0
        self.beta = 0.5

    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)

        # predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
        # targets = (targets - torch.mean(targets)) / torch.std(targets)

        targets = torch.nn.functional.normalize(targets, dim=1)
        predictions = torch.nn.functional.normalize(predictions, dim=1)

        l_time = self.timeLoss(predictions, targets)
        l_frequency = self.frequencyLoss(predictions, targets)
        return self.alpha * l_time + self.beta * l_frequency

    def frequencyLoss(self, predictions, target):
        batch, n = predictions.shape
        predictions = self.calculate_rppg_psd(predictions)
        target = self.calculate_rppg_psd(target)
        di = torch.log(predictions) - torch.log(target)
        sum_di_squared = torch.sum(di ** 2, dim=-1)
        sum_di = torch.sum(di, dim=-1)

        hybrid_loss = (1 / n) * sum_di_squared - (self.lambda_value / (n ** 2)) * sum_di ** 2
        loss = torch.sum(hybrid_loss) / batch
        return loss

    def calculate_rppg_psd(self, rppg_signal):
        spectrum = fft.fft(rppg_signal)
        # 복소 곱을 사용하여 PSD 계산
        psd = torch.abs(spectrum) ** 2

        return psd


class CurriculumLearningGuidedDynamicLoss(nn.Module):
    def __init__(self):
        super(CurriculumLearningGuidedDynamicLoss, self).__init__()
        # self.predicted_rppg, self.target_rppg, self.average_hr = predicted_rppg, target_rppg, average_hr
        self.fs, self.std = 30, 1.0
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()

        self.temporal_loss = 1.0
        self.batch_size = 0
        self.kl_loss = loss.KLDivLoss(reduction='batchmean', log_target=True)
        # self.complex_absolute = None

        self.init_alpha, self.init_beta = 0.1, 1.0
        self.alpha_exp, self.beta_exp = 0.5, 2.0
        self.alpha, self.beta = 0.05, 2.0

    def forward(self, epoch, predicted_rppg, target_ppg, average_hr):
        batch = predicted_rppg.shape[0]
        self.batch_size = batch
        if predicted_rppg.dim() == 1:
            predicted_rppg = predicted_rppg.view(1, -1)
            target_ppg = target_ppg.view(1, -1)

        # temporal_loss - neg Pearson correlation
        temporal_loss = neg_Pearson_Loss(predicted_rppg, target_ppg)
        complex_absolute = self.get_complex_absolute(predicted_rppg)
        cross_entropy_loss = self.calculate_frequency_loss(complex_absolute, average_hr)
        label_distribution_loss = self.calculate_label_distribution_loss(complex_absolute, average_hr)

        # curriculum learning setting
        if epoch > 25:
            self.alpha = 0.05
            self.beta = 2.0
        else:
            self.alpha = self.init_alpha * math.pow(self.alpha_exp, epoch / 25.0)
            self.beta = self.init_beta * math.pow(self.beta_exp, epoch / 25.0)
        # print('alpha: ', round(self.alpha, 3), 'beta: ', round(self.beta, 3))
        # print('temporal_loss: ', round(temporal_loss.item(), 3), 'cross_entropy_loss: ', round(cross_entropy_loss.item(), 3),
        #         'label_distribution_loss: ', round(label_distribution_loss.item(), 3))
        return self.alpha * temporal_loss + self.beta * (cross_entropy_loss + label_distribution_loss)
        # return  temporal_loss +  (cross_entropy_loss + label_distribution_loss)

    def get_complex_absolute(self, rppg):
        n = rppg.size()[1]
        unit_per_hz = self.fs / n
        feasible_bpm = self.bpm_range / 60.0
        k = (feasible_bpm / unit_per_hz).type(torch.FloatTensor).cuda().view(1, -1, 1)
        two_pi_n_over_N = (Variable(2 * math.pi * torch.arange(0, n, dtype=torch.float32),
                                    requires_grad=True) / n).cuda().view(1, 1, -1)
        hanning = (Variable(torch.from_numpy(np.hanning(n)).type(torch.FloatTensor),
                            requires_grad=True).view(1, -1)).cuda()
        hanning_rppg = (rppg * hanning).view(self.batch_size, 1, -1)  # (batch, 1, time length)

        complex_absolute = torch.sum(hanning_rppg * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 + \
                           torch.sum(hanning_rppg * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        # return F.softmax(complex_absolute, dim=-1)
        return (1.0 / complex_absolute.sum(keepdim=True, dim=-1)) * complex_absolute

    def calculate_frequency_loss(self, complex_absolute_softmax, hr):
        # frequency loss - cross entropy
        cross_entropy_loss = 0.0

        for b in range(self.batch_size):
            cross_entropy_loss += F.cross_entropy(complex_absolute_softmax[b].view(1, -1),
                                                  hr[b].view(1).type(torch.long))

        return cross_entropy_loss / self.batch_size

    def calculate_label_distribution_loss(self, softmax, hr):
        # frequency loss - label distribution
        label_distribution_loss = 0.0

        for b in range(self.batch_size):
            target_distribution = []
            for i in range(140):
                target_distribution.append(math.exp(-(i - int(hr[b])) ** 2 / (2 * self.std ** 2)) /
                                           (math.sqrt(2 * math.pi) * self.std))
            target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
            target_distribution = torch.Tensor(target_distribution).cuda()

            frequency_distribution = F.log_softmax(softmax[b].view(-1))
            label_distribution_loss += self.kl_loss(frequency_distribution, target_distribution)
            # print(self.kl_loss(frequency_distribution, target_distribution))

        return label_distribution_loss / self.batch_size


def peak_detection_loss(rppg, ppg, fs=30, epoch=15):
    ppg = ppg.view(5, -1)
    rppg = rppg.view(5, -1)
    ppg = normalize_torch(ppg)
    rppg = normalize_torch(rppg)
    test_n, sig_length = ppg.shape

    peak_score = 0.0
    hr_score = 0.0

    if epoch < 5:
        alpha = 1.0
        beta = 0.0
    elif epoch < 10:
        alpha = 0.5
        beta = 0.5
    else:
        alpha = 0.0
        beta = 1.0

    # ppg_hr_list = torch.empty(test_n)
    # hrv_list = torch.zeros((test_n, (sig_length // fs) * 3))
    # index_list = torch.zeros((test_n, (sig_length // fs) * 3))
    width = 11
    ppg_window_max = torch.nn.functional.max_pool1d(ppg, width, stride=1, padding=width // 2, return_indices=True)[
        1].squeeze()
    rppg_window_max = torch.nn.functional.max_pool1d(rppg, width, stride=1, padding=width // 2, return_indices=True)[
        1].squeeze()

    for i in range(test_n):
        ppg_candidate = ppg_window_max[i].unique()
        rppg_candidate = rppg_window_max[i].unique()
        ppg_nice_peaks = ppg_candidate[ppg_window_max[i][ppg_candidate] == ppg_candidate]
        ppg_nice_peaks = ppg_nice_peaks[ppg[i][ppg_nice_peaks] > torch.mean(ppg[i][ppg_nice_peaks]) / 2]
        # ppg_nice_peaks = ppg_nice_peaks[ppg_nice_peaks > 0.5] # if normalized from 0 to 1
        rppg_nice_peaks = rppg_candidate[rppg_window_max[i][rppg_candidate] == rppg_candidate]
        rppg_nice_peaks = rppg_nice_peaks[rppg[i][rppg_nice_peaks] > torch.mean(rppg[i][rppg_nice_peaks]) / 2]
        # rppg_nice_peaks = rppg_nice_peaks[rppg_nice_peaks > 0.5]
        # peak_diff = torch.abs(ppg_nice_peaks - rppg_nice_peaks)
        ppg_hrv = torch.diff(ppg_nice_peaks) / fs
        rppg_hrv = torch.diff(rppg_nice_peaks) / fs
        ppg_hr = torch.mean(60 / ppg_hrv)
        rppg_hr = torch.mean(60 / rppg_hrv)
        # peak_score += len(rppg_hrv) / len(ppg_hrv)
        peak_score += abs(len(ppg_hrv) - len(rppg_hrv)) / len(ppg_hrv)
        hr_score += abs(ppg_hr - rppg_hr) / ppg_hr

    # peak_score /= test_n
    hr_score /= test_n

    # return alpha * peak_score + beta * hr_score
    return hr_score


class PeakDetectionLoss(nn.Module):
    def __init__(self):
        super(PeakDetectionLoss, self).__init__()

    def forward(self, rppg, ppg, fs=30, epoch=15):
        return peak_detection_loss(rppg, ppg, fs, epoch)
