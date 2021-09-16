import numpy as np
import torch
import torchvision.transforms as transforms
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

class TTAWhiteboxProjectedGradientDescent(ProjectedGradientDescentPyTorch):
    attack_params = ProjectedGradientDescentPyTorch.attack_params + ["tta_size"]

    def __init__(
            self,
            estimator,
            norm: int = np.inf,
            eps: float = 0.3,
            eps_step: float = 0.1,
            max_iter: int = 100,
            targeted: bool = False,
            num_random_init: int = 0,
            batch_size: int = 32,
            random_eps: bool = False,
            tta_transforms: transforms.Compose = None,
            tta_size: int = 256):
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps)

        self.tta_size = tta_size
        self._check_params()
        self.tta_transforms = tta_transforms

    def _check_params(self) -> None:
        super()._check_params()
        if self.tta_size <= 0:
            raise ValueError("The number of tta samples `tta_size` has to be a positive integer.")

    def _compute_perturbation(self, x: "torch.Tensor", y: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = np.nan * torch.ones(x.size(), device=x.device)
        for i in range(len(x)):
            grad[i] = self.estimator.tta_loss_gradient_framework(x[i], y[i], self.tta_transforms, self.tta_size) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * mask
