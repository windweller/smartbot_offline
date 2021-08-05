import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        # This line causes error to be silent
        # while mask.dim() < vector.dim():
        #     mask = mask.unsqueeze(1)

        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    # return torch.nn.functional.log_softmax(vector, dim=dim)
    return torch.nn.functional.softmax(vector, dim=dim)

class MLPPolicy(nn.Module):
    def __init__(self, sizes, activation=nn.GELU,
                 output_activation=nn.Identity):
        super().__init__()
        self.nA = 4
        self.is_linear = False
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.net = nn.Sequential(*layers)

    def get_action_probability(self, obs, no_grad=True, action_mask=None, temperature=1):
        # obs: [batch_size, obs_dim]
        # we add a mask to forbid certain actions being taken
        if no_grad:
            with torch.no_grad():
                logits = self.net(obs)
                # mask here
                if action_mask is not None:
                    probs = masked_softmax(logits / temperature, action_mask)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
        else:
            logits = self.net(obs)
            # mask here
            if action_mask is not None:
                probs = masked_softmax(logits, action_mask)
            else:
                probs = F.softmax(logits, dim=-1)

        return probs

    def forward(self, obs):
        logits = self.net(obs)
        logp = F.log_softmax(logits, dim=-1)
        return logp