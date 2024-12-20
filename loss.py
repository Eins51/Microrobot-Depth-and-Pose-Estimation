import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss(alpha=0.2):
    def _triplet_loss(anchor, positive, negative):
        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).flatten()
        hard_triplets = torch.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    return _triplet_loss


class LabelSmoothSoftmaxCEV1(torch.nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            # >>> criteria = LabelSmoothSoftmaxCEV1()
            # >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            # >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            # >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def cosine_contrastive_loss(output1, output2, label):
    cosine_similarity = F.cosine_similarity(output1, output2)
    cosine_similarity = F.relu(cosine_similarity)
    loss = F.mse_loss(cosine_similarity, label.float())

    return loss


def cosine_contrastive_loss_with_margin(output1, output2, label, m=0.2):
    D = 1 - F.cosine_similarity(output1, output2)
    loss = 0.5 * label * (D ** 2) + 0.5 * (1 - label) * F.relu(m - D) ** 2
    return loss.mean()


def mix_loss(feat1, feat2, output1, output2, label1, label2):
    ccl = cosine_contrastive_loss(feat1, feat2, label1.eq(label2).float())
    ce1 = F.cross_entropy(output1, label1)
    ce2 = F.cross_entropy(output2, label2)
    loss = ccl + (ce1 + ce2) / 2
    return loss


def mix_loss_margin(m=0.9):
    def _mix_loss_margin(feat1, feat2, output1, output2, label1, label2):
        ccl = cosine_contrastive_loss_with_margin(feat1, feat2, label1.eq(label2).float(), m)
        ce1 = F.cross_entropy(output1, label1)
        ce2 = F.cross_entropy(output2, label2)
        loss = ccl + (ce1 + ce2) / 2
        return loss

    return _mix_loss_margin


def siam_ce_loss(feat1, feat2, output1, output2, label1, label2):
    ce1 = F.cross_entropy(output1, label1)
    ce2 = F.cross_entropy(output2, label2)
    loss = (ce1 + ce2) / 2
    return loss


def get_loss(name, margin=0.9):
    if name == "triplet":
        return triplet_loss(0.2)
    elif name == "mix":
        return mix_loss
    elif name == "ce":
        return siam_ce_loss
    elif name == "mix_m":
        return mix_loss_margin(margin)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    a = torch.from_numpy(np.array([1, 2, 3, 4]))
    b = torch.from_numpy(np.array([2, 2, 2, 4]))

    print(a.eq(b).int())
