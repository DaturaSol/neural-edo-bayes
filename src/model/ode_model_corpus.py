"""
Module containing GRU ODE Bayes main body cell.
The model is defined here, this is a simplified version of the GRU-ODE-Bayes model
provide in the original repository. I have done quite a few changes to the original code
to make it more readable and understandable. Also, have include modern tatichs to parallelize the code
and make it more efficient. Since this model deals with ODE solvers, one of the main chalanges is parelalizing the code.
My atempt is to make it as efficient as possible while keeping the code readable and understandable.
"""

# src.model.ode_model_corpus

import torch
from torch import nn, Tensor
from torchdiffeq import (  # Importing the ODE solvers from torchdiffeq
    odeint,  # NOTE: The original source uses odeint, but I have changed it to odeint_adjoint.
    odeint_adjoint,  # The adjoint method is more efficient for memory.
)
from typing import Any, Optional, Literal

# NOTE: I will not be implementing the autonomous cells,
# since they are not used in the current implementation.
from src.model.cells import (
    GRUODECell,
    FullGRUODECell,
    PModelLogvar,
    GRUObservationCellLogvar,
)
from src.model.aux_func import init_weights, compute_KL_loss


# NOTE: The original repository implements in python there own ODE solvers,
# which is a performance killer, so I will be using the torchdiffeq library.
# It is most likely that our EDO will be stiff for a complex system,
# so i do not recommend using small step sizes, solvers as dopri8 or dopri5.
# From my experience, the best solvers for stiff systems are:
# rk4, euler, midpoint, heun2, heun3, explicit_adams, implicit_adams.
solvers = Literal[
    "dopri8",
    "dopri5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
    "euler",
    "midpoint",
    "heun2",
    "heun3",
    "rk4",
    "explicit_adams",
    "implicit_adams",
]


class GRUODEBayes(nn.Module):
    """
    Very basic implementation of the GRU-ODE-Bayes model.
    With some modifications to make it more efficient and readable.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_hidden_size: int,
        prep_hidden_size: int,
        cov_size: int,
        cov_hidden_size: int,
        class_hidden_size: int,
        class_output_size: int,
        bias: bool = True,
        dropout_rate: float = 0.0,
        mixing: float = 1.0,
        full_gru_ode: bool = True,
        solver: solvers = "euler",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        fixed_step_size: bool = False,
    ) -> None:
        super().__init__()

        self.class_model = nn.Sequential(
            nn.Linear(hidden_size, class_hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(class_hidden_size, class_output_size, bias=bias),
        )  # Classifier Model, predict the class of the input data.

        self.cov_model = nn.Sequential(
            nn.Linear(cov_size, cov_hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(cov_hidden_size, hidden_size, bias=bias),
            nn.Tanh(),
        )  # Covariates Model, produces hidden state based on initial static data.

        self.p_model = PModelLogvar(
            hidden_size, p_hidden_size, input_size, bias=bias
        )  # Probability Model, produces the mean and logvar, for a given hidden state.

        self.gru_obs = GRUObservationCellLogvar(
            input_size, hidden_size, prep_hidden_size, bias=bias
        )  # GRU Observation Model, produces a new hidden state and loss,
        # based on a new observation.

        # NOTE: I gave myself the freedom to feed the full probabilty distribution for this
        # this diverges from the original source.
        if full_gru_ode:
            self.gru_c = FullGRUODECell(
                input_size * 2, hidden_size, bias=bias
            )  # GRU dynamics to be used in the ODE
        else:
            self.gru_c = GRUODECell(
                input_size * 2, hidden_size, bias=bias
            )  # GRU dynamics to be used in the ODE

        # Defined at `__init__` to save resources on forward method.
        def dynamics(t: Tensor, h_local: Tensor) -> Tensor:
            """
            Wrapper for the dynamics of our system.
            * `odeint_adjoint` expects a parameter for time.
            This is mostly to keep the logic on the cells module organized, but
            could very well be implemented there.
            """
            p = self.p_model(h_local)
            dh = self.gru_c(p, h_local)
            return dh

        self.adjoint_params = list(self.p_model.parameters()) + list(
            self.gru_c.parameters()
        )  # Also defined at `__init__` to save resources
        self.dynamics = dynamics
        self.solver = solver
        self.mixing = mixing
        self.rtol = rtol
        self.atol = atol
        self.fixed_step_size = fixed_step_size

        self.apply(init_weights)

    def forward(
        self,
        times: Tensor,
        time_ptr: Tensor,
        X: Tensor,
        M: Tensor,
        obs_idx: Tensor,
        obs_to_time_idx: Tensor,
        cov: Tensor,
        T: Tensor,
        labels: Tensor,
        class_criterion: nn.Module,
        class_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # NOTE: This is my own implemetation which focuss on trying to parellize this,
        # this diverges a loot from the original purposed method.
        # Reasons as to why, is that training with anything close to parallel given the original
        # configuration is as infuriating as a kidnee stone.
        # A brief explanation....
        # The original source is very intuitive, the process goes:
        # For all observed values: [New observation -> ODE Solver -> New Prediciton -> Observation Jump -> ...]
        # After all observations are done, it propagates the EDO to the final Time.
        # The observation values occuours inside a python loop, this means that we are victims
        # to the ode solver, which is computationally heavy, and requires some back and fourth from the GPU to CPU.
        # I am not quite sure of how `pytorchdiff` works in the backend, quite sure it uses C++ to match pytorch.
        # Beyond all we need to feed a `pytorch.Tensor` to the ode solver which carries a loot of information,
        # this creates a very slow and gradual ascent in memory consumption, which is a tall tail sign that it is not
        # properly allocating memory and working in parallel.
        # HACK: The solution that i am to apply is to vectorize all and only do one single ODE solver for all the time,
        # Since this is only the forward step and no update to the gradient ever occours.
        # The only problem is that for the ODE solver the initial states which are supposed
        # to change accordinly to every update now will be feed only once.
        # This means that the hidden state is actually never updated, and needs to be
        # actually updated in the next forward pass.

        h = self.cov_model(cov)  # Initial guess given static parameters.

        # All times for the EDO Solver to iterate at once
        # NOTE: Normalizing the time scale gave repeted zero values...
        t_all = torch.cat([torch.tensor([0.0], device=times.device), times])
        t_all = torch.unique(t_all)
        # --- DEBUGGING BLOCK ---
        assert torch.all(torch.diff(t_all) > 0), "t_all is not strictly increasing!"
        # ----
        options = {}

        if (self.solver in ["euler", "rk4", "midpoint"]) and self.fixed_step_size:
            if len(t_all) > 1:
                min_diff = torch.diff(t_all).min()
                step_size = max(min_diff.item() / 2.0, 1e-3)
                options["step_size"] = step_size
            else:
                options["step_size"] = 0.1

        # --- Calls ODE Solver ---
        # Gets the hidden state at every unique observation time point,
        # for every sample in the batch...
        h_all_times = odeint_adjoint(
            self.dynamics,
            h,
            t_all,
            adjoint_params=self.adjoint_params,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            options=options if options else None,
        )  # Shape: [num_times, batch_size, hidden_size]

        # --- State *before* each observation jump ---
        # We need the state *before* the jump. If an observation is at `times[k]`,
        # its state is in `h_all_times` at the index corresponding to `times[k]`.
        # `t_all` has `t=0` at index 0, so `times[k]` corresponds to `h_all_times[k+1]`.
        # If `times` already starts with 0, then `times[k]` is at `h_all_times[k]`.

        time_indices_for_h = obs_to_time_idx + (
            1 if t_all[0] == 0.0 and times[0] != 0.0 else 0
        )

        # This is the clean, robust way to select the states.
        # "For each observation, get its time_index and its patient_index, and select."
        h_pre_jump = h_all_times[  # type: ignore
            time_indices_for_h, obs_idx
        ]  # Shape: [num_t_all, batch_size, hidden_size]

        p_pre_jump = self.p_model(
            h_pre_jump
        )  # Predicted distribution parameters at these states

        # Perform the discrete update step for ALL observations at once
        h_post_jump, loss_recon = self.gru_obs(
            h_pre_jump, p_pre_jump, X, M
        )  # Shape: [num_obs_total, hidden_size]

        # --- Calculates Loss ---
        # A. KL Divergence Loss
        p_post_jump = self.p_model(h_post_jump)
        loss_kl = compute_KL_loss(p_obs=p_post_jump, X_obs=X, M_obs=M)
        # B. Per-Observation Classification Loss
        pred_obs = self.class_model(h_post_jump)
        labels_obs = labels[obs_idx]
        total_class_loss = class_criterion(pred_obs, labels_obs).sum()

        # --- Construct the Finall hidden state ---
        # NOTE: We need to account for all the jumps.
        jump_deltas = h_post_jump - h_pre_jump
        # We create a "jump matrix" that will sum up all deltas for each patient in the batch.
        h_jump_total = torch.zeros_like(h)  # Shape: [batch_size, hidden_size]
        # `index_add_` is a powerful scatter-add operation. It adds `jump_deltas` to
        # the rows of `h_jump_total` specified by `obs_idx`.
        h_jump_total.index_add_(0, obs_idx, jump_deltas)

        # The final state is the continuous evolution to T, plus the sum of all jumps.
        # It assumes the jumps are small and their effect on the final state is additive.
        h_final_continuous = h_all_times[-1]  # type: ignore

        # Propagate the final continuous state to time T
        if T > t_all[-1]:
            t_eval_final = torch.stack([t_all[-1], T])
            h_final_cont_T = odeint_adjoint(
                self.dynamics,
                h_final_continuous,
                t_eval_final,
                adjoint_params=self.adjoint_params,
                rtol=self.rtol,
                atol=self.atol,
                options=options if options else None,
            )
            h_final_continuous = h_final_cont_T[-1]  # type: ignore

        # Add the summed jumps to the final continuous state
        h_final = (
            h_final_continuous + h_jump_total
        )  # type: ignore # A trick to preserve gradients through the continuous path
        h_final = h_final + h_jump_total

        # --- Final Loss Calculation at T ---
        final_class_pred = self.class_model(h_final)
        final_class_loss = class_criterion(final_class_pred, labels).sum()
        total_class_loss = total_class_loss + final_class_loss

        # --- Combine all losses ---
        total_loss = (
            loss_recon + self.mixing * loss_kl + total_class_loss * class_weight
        )

        return h_final, total_loss, final_class_pred, total_class_loss

    # NOTE: Make this the forward method for the BASELINE
    def forward_(
        self,
        times: Tensor,
        time_ptr: Tensor,
        X: Tensor,
        M: Tensor,
        obs_idx: Tensor,
        obs_to_time_idx: Tensor,
        cov: Tensor,
        T: Tensor,
        labels: Tensor,
        class_criterion: nn.Module,
        class_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        DUMMY forward pass for performance profiling.
        This version bypasses the expensive odeint calls to isolate their impact.
        """
        # --- This part is the same ---
        h = self.cov_model(cov)

        t_all = torch.unique(
            torch.cat([torch.tensor([0.0], device=times.device), times])
        )

        # --- DUMMY BYPASS FOR THE MAIN ODE SOLVE ---
        # Instead of calling odeint, we create a placeholder tensor.
        # We just repeat the initial hidden state `h` for each time step.
        # This is computationally cheap but has the correct shape for the rest of the code.
        num_times = len(t_all)
        # Shape: [num_times, batch_size, hidden_size]
        h_all_times = h.unsqueeze(0).expand(num_times, -1, -1)

        # --- The rest of the logic can now run, using the dummy tensor ---

        # Advanced Indexing to get the state before each observation
        time_indices_for_h = obs_to_time_idx + (
            1 if t_all[0] == 0.0 and times[0] != 0.0 else 0
        )
        h_pre_jump = h_all_times[time_indices_for_h, obs_idx]

        # Discrete update and loss calculation
        p_pre_jump = self.p_model(h_pre_jump)
        h_post_jump, loss_recon = self.gru_obs(h_pre_jump, p_pre_jump, X, M)

        p_post_jump = self.p_model(h_post_jump)
        loss_kl = compute_KL_loss(p_obs=p_post_jump, X_obs=X, M_obs=M)

        pred_obs = self.class_model(h_post_jump)
        labels_obs = labels[obs_idx]
        total_class_loss = class_criterion(pred_obs, labels_obs).sum()

        # Reconstruct the "jump" part of the final hidden state
        # NOTE: In a dummy run, jump_deltas will be based on a constant `h_pre_jump`,
        # so this part won't be very meaningful, but it runs.
        jump_deltas = h_post_jump - h_pre_jump.detach()  # Detach here
        h_jump_total = torch.zeros_like(h)
        h_jump_total.index_add_(0, obs_idx, jump_deltas)

        # --- DUMMY BYPASS FOR THE FINAL PROPAGATION ---
        # We just use the state from the end of the observation window.
        h_final_continuous = h_all_times[-1]

        # The final state is approximated as the continuous part + jumps
        h_final = h_final_continuous + h_jump_total

        # --- Final Loss Calculation ---
        final_class_pred = self.class_model(h_final)
        final_class_loss = class_criterion(final_class_pred, labels).sum()
        total_class_loss = total_class_loss + final_class_loss

        total_loss = (
            loss_recon + self.mixing * loss_kl + total_class_loss * class_weight
        )

        return h_final, total_loss, final_class_pred, total_class_loss
