import torch
from torch.nn import functional as F
import numpy as np 

def JSD(out1, out2):
    """
    Jensen-Shannon divergence (symmetric) between two outputs
    """
    prob1 = F.softmax(out1, dim=1)
    prob2 = F.softmax(out2, dim=1)
    
    m = 0.5 * (prob1 + prob2)
    
    kl_pm = F.kl_div(F.log_softmax(out1, dim=1), m, reduction='batchmean')
    kl_qm = F.kl_div(F.log_softmax(out2, dim=1), m, reduction='batchmean')

    return 0.5 * (kl_pm + kl_qm)


def debias(current_logit, qhat, tau=0.5):
    """
    debias logits, then return probabilities
    """
    debiased_prob = F.softmax(current_logit - tau * torch.log(qhat), dim=1)
    return debiased_prob

def update_qhat(probs, qhat, momentum, qhat_mask=None):
    """
    update qhat of class prior with momentum
    """
    if qhat_mask is not None:
        mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat


def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True) -> torch.Tensor:
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y),device=y.device),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac = Jac.reshape(shape=(y.shape + x.shape))
    return Jac

def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,need_higher_grad = True) -> torch.Tensor:
    sumed_y = batched_y.sum(dim = 0)
    J = jacobian(sumed_y, batched_x, need_higher_grad)
    
    dims = list(range(J.dim()))
    dims[0],dims[sumed_y.dim()] = dims[sumed_y.dim()],dims[0]
    J = J.permute(dims = dims)
    return J

def grad_cam(c, feat):
    grad = batched_jacobian(c, feat).to(feat.device)
    grad_cam = F.relu_((grad.mean(2, keepdim = True) * feat.unsqueeze(1)).sum(-1))
    return  grad_cam[..., 1:]

def ppmcc(source_maps, target_maps, rho = 0.5):
    """
    pairwise pearson matrix correlation coefficient
    """
    M = source_maps.shape[0]
    N = target_maps.shape[0]
    
    correlation_matrix = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            correlation_matrix[i, j] = np.corrcoef(source_maps[i,:].detach().cpu().numpy(), target_maps[j,:].detach().cpu().numpy())[0, 1]
            if np.isnan(correlation_matrix[i, j]):
                correlation_matrix[i, j] = 0.0

    if np.all(np.any(correlation_matrix > rho, axis=1)):
        flat_ind = np.argsort(correlation_matrix.ravel())[::-1]
        row_inds, col_inds = np.unravel_index(flat_ind, correlation_matrix.shape)
        max_col = [-1] * M
        used_rows = set()
        used_cols = set()

        for row_idx, col_idx in zip(row_inds, col_inds):
            if row_idx not in used_rows and col_idx not in used_cols:
                max_col[row_idx] = col_idx
                used_rows.add(row_idx)
                used_cols.add(col_idx)
            if len(used_rows) == M:
                break
        max_col = torch.tensor(max_col, dtype=torch.int)
        return max_col
    
    else:
        flat_ind = np.argsort(correlation_matrix.ravel())[::-1]
        row_inds, col_inds = np.unravel_index(flat_ind, correlation_matrix.shape)
        max_col = []
        used_cols = set()
        for row_idx, col_idx in zip(row_inds, col_inds):
            if col_idx not in used_cols:
                max_col.append(col_idx)
                used_cols.add(col_idx)
            if len(max_col) == M:
                break
        max_col = torch.tensor(max_col, dtype=torch.int)
        return max_col