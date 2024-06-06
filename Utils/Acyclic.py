import torch
import numpy as np
from scipy.linalg import schur

class Acyclic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        D = input
        E, U = schur((D).cpu().detach().numpy())

        h = 0.5 * np.sum(np.diag(E) ** 2)

        E = torch.from_numpy(E).to(input.device)
        U = torch.from_numpy(U).to(input.device)
        ctx.save_for_backward(D, E, U)
        return torch.as_tensor(h, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        D,E,U = ctx.saved_tensors
        G_h = U.matmul(torch.diag(torch.diag(E.t()))).matmul(U.T)
        grad_input = grad_output * G_h * torch.sqrt(D) * 2
        return grad_input


acyclic = Acyclic.apply
