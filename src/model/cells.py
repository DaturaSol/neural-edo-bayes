"""
Module containing cells (`nn.Module`) used in the model.
----

---

Includes:
-----
1. ObservableCell:
    Which is a custom cell for handling observations, discrete updates based on the received observations.
2. PModel:
    Which is a custom wraper for the probability predictor model. Using the logvar option for KL loss.
    We restrict the logvar to be in a certain range to avoid numerical issues.
3. FullGRUODECell_Autonomous:
    Defines the autonomous dynamics of the hidden state h.
4. FullGRUODECell:
    Defines the non-autonomous, input-driven dynamics of the hidden state h.
5. GRUODECell_Autonomous:
    Defines the autonomous dynamics for a simplified GRU-ODE cell. Lacks a reset gate (r).
6. GRUODECell:
    Defines the input-driven dynamics for a simplified GRU-ODE cell. Lacks a reset gate (r).
    
NOTE: Autonomous cells are used when no external input is provided.
"""

# src.model.cells

import torch
import numpy as np
from torch import nn, Tensor


class GRUObservationCellLogvar(nn.Module):
    """
    Implements GRU cell responsible for handling discrete updates based on the received observations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        prep_hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden_size * input_size, hidden_size, bias=bias)

        std = np.sqrt(2.0 / (4 + prep_hidden_size))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 3, prep_hidden_size))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden_size))

        self.input_size = input_size
        self.prep_hidden_size = prep_hidden_size

    def forward(
        self, h_obs: Tensor, p_obs: Tensor, X_obs: Tensor, M_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Processes a batch of observations to update their corresponding hidden states.
        Returns hidden state and loss.
        """
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)  # Standard deviation from log variance
        error = (X_obs - mean) / sigma

        loss = 0.5 * ((torch.pow(error, 2) + logvar) * M_obs).sum()

        gru_input = torch.stack([mean, logvar, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (
            (gru_input * M_obs)
            .permute(1, 2, 0)
            .contiguous()
            .view(-1, self.prep_hidden_size * self.input_size)
        )

        h_obs_new = self.gru_d(gru_input, h_obs)
        return h_obs_new, loss


class PModelLogvar(nn.Module):
    """
    The p(t) model that takes h(t) and returns the distribution parameters.
    It includes a Tanh activation ONlY on the log-variance for stability,
    while leaving the mean unconstrained to match the normalized data.
    """

    def __init__(
        self, hidden_size: int, p_hidden_size: int, input_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, p_hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(p_hidden_size, 2 * input_size, bias=bias),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw_output = self.net(h)
        mean, logvar_raw = torch.chunk(raw_output, 2, dim=-1)

        # Stabilize logvar by bounding it to a reasonable range.
        logvar_stabilized = torch.tanh(logvar_raw) * 5

        return torch.cat([mean, logvar_stabilized], dim=-1)


class FullGRUODECell(nn.Module):
    r"""
    Defines the non-autonomous, input-driven dynamics of the hidden state h.

    This cell is used to evolve the hidden state h(t) continuously through
    time when the model's own predictions are used as a continuous input
    (i.e., between observations and when the 'impute' flag is True).
    The change in h is a function of both h and an external input x,
    which represents the predicted observation parameters p: dh/dt = f(h, x).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.lin_x = nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Executes one step with GRU-ODE for all h."""
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(h * r))

        dh = (1 - z) * (u - h)
        return dh  # Returns the dynamics of system


class FullGRUODECell_Autonomous(nn.Module):
    r"""
    Defines the autonomous dynamics of the hidden state h.

    This cell is used to evolve the hidden state h(t) continuously through
    time when no external input is provided (i.e., between observations
    and when the 'impute' flag is False). The change in h is a function
    of h alone: dh/dt = f(h).
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """Executes one step with autonomous GRU-ODE for all h."""
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh  # Returns the dynamics for our ODE


class GRUODECell(nn.Module):
    r"""
    Defines the input-driven dynamics for a simplified GRU-ODE cell.

    This cell is based on a "Minimal GRU" which lacks a reset gate (r).
    It models the evolution of the hidden state :math:`h(t)` when driven
    by a continuous external input :math:`x(t)`. In the context of the
    GRU-ODE-Bayes model, this input is typically the model's own prediction
    of the data parameters, used when 'impute' is ``True``.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()

        self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Computes the derivative of the hidden state for the ODE solver."""
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)

        return dh


class GRUODECell_Autonomous(nn.Module):
    r"""
    Defines the autonomous dynamics for a simplified GRU-ODE cell.

    This cell is based on a "Minimal GRU" which lacks a reset gate (r).
    It models the evolution of the hidden state :math:`h(t)` when no
    external input is provided (e.g., between observations when the 'impute'
    flag is set to ``False``).
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """Computes the derivative of the hidden state for the ODE solver."""
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh
