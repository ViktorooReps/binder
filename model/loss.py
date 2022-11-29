import torch
from torch import Tensor, LongTensor, logsumexp
from torch.nn import Module

REDUCTION = {
    'mean': torch.nanmean,
    'sum': torch.nansum,
    'none': lambda x, dim: x
}


class ContrastiveThresholdLoss(Module):

    def __init__(self, n_classes: int, ignore_id: int, reduction: str, beta: float):
        super(ContrastiveThresholdLoss, self).__init__()
        self._beta = beta
        self._n_classes = n_classes
        self._ignore_id = ignore_id
        self._reduce = REDUCTION[reduction]

    def forward(self, predicted_scores: Tensor, true_ids: LongTensor) -> Tensor:
        """Computes positive-unlabelled loss.

        :param predicted_scores: (B, S, N, C)
        :param true_ids:  (B, S, N)
        :return: loss with applied reduction
        """
        assert (true_ids[:, 0, 0] == self._ignore_id).all()  # the first label should be [CLS]
        device = predicted_scores.device
        classes = torch.arange(self._n_classes, device=device)

        ignore_mask = (true_ids == self._ignore_id).unsqueeze(1).repeat(1,  self._n_classes, 1, 1)  # (B, C, S, N)
        class_mask = (classes.reshape(1,  self._n_classes, 1, 1) == true_ids.unsqueeze(1))  # (B, C, S, N)

        predicted_scores = predicted_scores.swapaxes(-2, -1).swapaxes(-3, -2)  # (B, S, N, C) -> (B, C, S, N)

        denominator_mask = (~class_mask & ~ignore_mask)  # elems for denominator
        denominator_mask[:, :, 0, 0] = True  # include cls score

        predicted_scores = predicted_scores.swapaxes(-2, -1).swapaxes(-3, -2)  # (B, S, N, C) -> (B, C, S, N)

        denominator_score = torch.clone(predicted_scores)
        denominator_score[~denominator_mask] = -torch.inf  # exp will turn this into 0
        denominator_score = logsumexp(denominator_score, dim=[-2, -1])  # (B, C)
        cls_score = denominator_score - predicted_scores[:, :, 0, 0]

        # mean over positives
        positive_scores_mask = (~ignore_mask & class_mask)
        predicted_scores[~positive_scores_mask] = torch.nan
        predicted_scores = predicted_scores.nanmean(dim=[-2, -1])

        # use [CLS] label as a threshold
        contrastive_losses = denominator_score - predicted_scores
        threshold_losses = contrastive_losses * self._beta + (1 - self._beta) * cls_score

        return self._reduce(threshold_losses)
