import torch
from pesq import pesq


class SEMetric(torch.nn.Module):
    def __init__(
        self, metric: str, sampling_rate: int, reduction: str = "mean", sgn: int = 1
    ):
        super().__init__()
        self.metric = metric
        self.sampling_rate = sampling_rate
        self.reduction = reduction
        self.sgn = sgn
        assert self.sgn in [1, -1]

    def forward(self, ref, est):
        if self.metric == "sisdr":
            scores = self.sisdr(ref, est) * self.sgn
        elif self.metric == "pesq":
            scores = self.pesq(ref, est) * self.sgn
        else:
            raise ValueError("metric must be in [sisdr, pesq]")
        if self.reduction == "mean":
            return torch.mean(scores)
        elif self.reduction == "sum":
            return torch.sum(scores)
        elif self.reduction == "none":
            return scores
        else:
            raise ValueError("metric must be in [mean, sum, none]")

    def _safe_db(self, num, den):
        if 0 in den:
            return torch.Tensor([torch.inf])
        return 10 * torch.log10(num / den)

    def sisdr(self, ref, est):
        """Returns a SI-SDR score
        Args:
            ref (tensor): clean speech. (..., n_time)
            est (tensor): estimated signal. (..., n_time)
        Returns:
            sisdr_val (tensor): score of the SI-SDR
        """
        assert est.shape == ref.shape
        energy_s_true = torch.sum(ref**2, dim=-1, keepdim=True)
        lamb = torch.sqrt(energy_s_true / torch.sum(est**2, dim=-1, keepdim=True))
        distortion = torch.sum((ref - lamb * est) ** 2, dim=-1, keepdim=True)
        sisdr_val = self._safe_db(energy_s_true, distortion)
        return sisdr_val

    def pesq(self, ref, est):
        """Returns a PESQ score
        Args:
            ref (tensor): clean speech. (..., n_time)
            est (tensor): estimated signal. (..., n_time)
        Returns:
            PESQ: Mean score of the PESQ
        """
        assert est.shape == ref.shape
        batch_shape = ref.shape[:-1]
        ref = ref.flatten(end_dim=-2).to("cpu").detach().numpy().copy()
        est = est.flatten(end_dim=-2).to("cpu").detach().numpy().copy()
        pesq_val = torch.zeros(len(ref))
        # total_score = 0
        for i in range(len(ref)):
            pesq_val[i] = pesq(self.sampling_rate, ref[i], est[i], "wb")
        return pesq_val.reshape(batch_shape)
