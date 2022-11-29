import torch
from torch import Tensor, LongTensor, BoolTensor
from torch.nn import Module

REDUCTION = {
    'mean': torch.nanmean,
    'sum': torch.nansum,
    'none': lambda x, dim: x
}


def masked_logsumexp_w_elem(tensor: Tensor, mask: BoolTensor):
    masked_tensor = torch.clone(tensor)
    masked_tensor[~mask] = -torch.inf
    maxes = torch.amax(masked_tensor, [-1, -2], keepdim=True)
    masked_exp_tensor = torch.exp(masked_tensor - maxes)
    exp_tensor = torch.exp(tensor - maxes)
    sum_w_elem = torch.sum(masked_exp_tensor, [-1, -2], keepdim=True)
    return sum_w_elem.log().add(maxes)


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

        denominator_mask = (~class_mask & ~ignore_mask)  # elems for denominator
        denominator_mask[:, :, 0, 0] = True  # include cls score

        predicted_scores = predicted_scores.swapaxes(-2, -1).swapaxes(-3, -2)  # (B, S, N, C) -> (B, C, S, N)

        denominator_score = masked_logsumexp_w_elem(predicted_scores, denominator_mask)  # (B, C, S, N)
        contrastive_score = denominator_score - predicted_scores
        cls_score = contrastive_score[:, :, 0, 0]

        print(f'cs: {torch.isnan(contrastive_score).sum()}/{torch.numel(contrastive_score)}')

        # mean over positives
        positive_scores_mask = (~ignore_mask & class_mask)
        contrastive_score[~positive_scores_mask] = torch.nan
        contrastive_losses = contrastive_score.nanmean(dim=[-2, -1])

        print(f'cl: {torch.isnan(contrastive_losses).sum()}/{torch.numel(contrastive_losses)}')

        # use [CLS] label as a threshold
        threshold_losses = contrastive_losses * self._beta + (1 - self._beta) * cls_score

        return self._reduce(threshold_losses)
