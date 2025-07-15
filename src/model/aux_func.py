"""
Module necesasry for auxiliary functions used in the model.
Includes weight initialization and KL divergence calculations.
The original repository in which this is based includes a non logvar option for the KL loss,
but it is not used in the current implementation.
The vanishing gradient problem for this model is solved partially by using the logvar option.
"""

# # src.model.aux_func

import torch
from torch import nn, Tensor


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


def compute_KL_loss(
    p_obs: Tensor,
    X_obs: Tensor,
    M_obs: Tensor,
    obs_noise_std: float = 1e-2,
) -> Tensor:
    r"""
    Calculates the Kullback-Leibler Divergence:
    ---
    :math:`D_{KL}\left( P || Q \right) = \sum_{x \in X} P(x)\ \log\left( \frac{P(x)}{Q(x)} \right)`

    ---

    Is a type of statistical distance:
    A measure of how much a model probability distribution
    :math:`Q` is different from a true probability distribution :math:`P`.
    """
    obs_noise_std_tensor = torch.tensor(obs_noise_std)
    mean, logvar = torch.chunk(p_obs, 2, dim=1)
    std = torch.exp(0.5 * logvar)

    return (
        gaussian_KL(
            mu_1=mean,
            mu_2=X_obs,  # We consider the new observed value to be the new Expected value
            sigma_1=std,
            sigma_2=obs_noise_std_tensor,  # Set a very small starndart deviation for it
        )
        * M_obs  # Only update the observed ones.
    ).sum()


def gaussian_KL(mu_1: Tensor, mu_2: Tensor, sigma_1: Tensor, sigma_2: Tensor) -> Tensor:
    r"""
    Calculates the Kullback-Leibler Divergence:
    ---
    :math:`D_{KL}\left( P || Q \right) = \sum_{x \in X} P(x)\ \log\left( \frac{P(x)}{Q(x)} \right)`

    For a Gaussian Distribution:
    :math:`P(X | \mu,\ \sigma) = \frac{1}{\sigma\ \sqrt{2\pi}}\ e^{-\frac{1}{2}\ \left( \frac{X - \mu}{\sigma} \right)^2}`

    ---

    The calculations for this is pretty complex so i will just input:
    :math:`D_{\text{KL}}\left({\mathcal {p}}\parallel {\mathcal {q}}\right)=\log {\frac {\sigma _{1}}{\sigma _{0}}}+{\frac {\sigma _{0}^{2}+{\left(\mu _{0}-\mu _{1}\right)}^{2}}{2\sigma _{1}^{2}}}-{\frac {1}{2}}`

    More on this proof can be found in the pdf which is include with this repository.
    """

    return (
        torch.log(sigma_2)
        - torch.log(sigma_1)
        + (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2))
        / (2 * torch.pow(sigma_2, 2))
        - 0.5  # I am tempted on removing this guy...
        # Once the gradient is calculated it will just vanish anyways.
    )
