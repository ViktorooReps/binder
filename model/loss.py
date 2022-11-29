import torch
from torch import Tensor, LongTensor, logsumexp
from torch.nn import Module

REDUCTION = {
    'mean': torch.mean,
    'sum': torch.sum,
    'none': lambda x, dim: x
}


class ContrastiveThresholdLoss(Module):

    def __init__(self, n_classes: int, unk_id: int, ignore_id: int, reduction: str, beta: float):
        super(ContrastiveThresholdLoss, self).__init__()
        self._beta = beta
        self._n_classes = n_classes
        self._unk_id = unk_id
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

        ignore_mask = (true_ids == self._ignore_id).unsqueeze(1)  # (B, 1, S, N)
        unk_mask = (true_ids == self._unk_id).unsqueeze(1)  # (B, 1, S, N)
        class_mask = (classes.reshape(1,  self._n_classes, 1, 1) == true_ids.unsqueeze(1))  # (B, C, S, N)
        denominator_mask = ((~class_mask | unk_mask) & ~ignore_mask)

        predicted_scores = predicted_scores.swapaxes(-2, -1).swapaxes(-3, -2)  # (B, S, N, C) -> (B, C, S, N)

        denominator_score = torch.clone(predicted_scores)
        denominator_score[denominator_mask] = -torch.inf  # exp will turn this into 0
        denominator_score = logsumexp(denominator_score, dim=[-2, -1])  # (B, C)

        positive_scores_mask = (~unk_mask & ~ignore_mask).repeat(1, self._n_classes, 1, 1)
        predicted_scores[positive_scores_mask] = torch.nan
        print(denominator_score.mean(), torch.nanmean(predicted_scores, dim=(-1, -2)).mean())

        # use [CLS] label as a threshold
        contrastive_losses = denominator_score.unsqueeze(-1).unsqueeze(-1) - predicted_scores
        threshold_losses = contrastive_losses * self._beta - (1 - self._beta) * contrastive_losses[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)

        # select only positive scores
        positive_scores_mask = (~unk_mask & ~ignore_mask).repeat(1, self._n_classes, 1, 1)
        return self._reduce(threshold_losses[positive_scores_mask])
