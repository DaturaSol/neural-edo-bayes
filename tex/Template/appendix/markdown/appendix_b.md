
### **Appendix B: Generalized Adjoint Method via Hamiltonian Formalism**

Building upon the previous derivation, we can generalize the adjoint method for more complex models by employing a Hamiltonian framework. This approach, inspired by optimal control theory, offers a more structured and extensible formulation, particularly for models like GRU-ODE-Bayes that may include a running loss term.

Let's define a total loss functional $T_L$ that includes both a final loss and an integrated loss over the trajectory:
$$
T_L(\vec{q}_n, \theta, t_1, t_0) = L(\vec{q}_n(t_1)) + \int_{t_0}^{t_1} \mathcal{L}(\vec{q}_n(t), \theta, t)\ dt
$$
Here, $\vec{q}_n(t)$ represents the state of the system, and $\mathcal{L}$ is a loss density (e.g., a regularization term). The system is subject to the dynamic constraint:
$$
\dot{\vec{q}}_n(t) = f_n(\vec{q}_i(t), \theta, t)
$$
and the boundary condition $\delta\vec{q}_n(t_0) = 0$, as the initial state is fixed. For the following, we will use the Einstein summation convention over repeated indices.

We define an augmented functional $J$ using Lagrange multipliers $\vec{p}_n(t)$ (the co-state or adjoint variables):
$$
J(\vec{q}_n, \vec{p}_n, \theta) = L(\vec{q}_n(t_1)) + \int_{t_0}^{t_1} \mathcal{L}(\vec{q}_n, \theta, t)\ dt
+ \int_{t_0}^{t_1} \vec{p}_i(t) \cdot \left(
\dot{\vec{q}}_i(t) - f_i(\vec{q}_i, \theta, t)
\right)\ dt
$$
We now define the **Hamiltonian**, $H$, for this system:
$$
H(\vec{q}_n, \vec{p}_n, \theta, t) = \vec{p}_i(t) \cdot f_i(\vec{q}_i, \theta, t) - \mathcal{L}(\vec{q}_n, \theta, t)
$$
This allows us to rewrite the functional $J$ more compactly:
$$
J(\vec{q}_n, \vec{p}_n, \theta)  = L(\vec{q}_n(t_1))
+ \int_{t_0}^{t_1} \left[
\vec{p}_i(t) \cdot \dot{\vec{q}}_i(t) -  H(\vec{q}_n, \vec{p}_n, \theta, t)
\right]\ dt
$$
We now compute the first variation of $J$. If we choose our adjoint variables $\vec{p}_n$ such that the variations with respect to $\vec{q}_n$ and $\vec{p}_n$ are zero, then the variation of $J$ will only depend on the explicit variation of $\theta$.

#### **B.1 First Variation of the Functional**

##### **Variation with respect to $\vec{q}_n$**
$$
\delta J_{\vec{q}_n} = \frac{\partial L}{\partial \vec{q}_n(t_1)} \cdot \delta \vec{q}_n(t_1)  +
\int_{t_0}^{t_1} \left[
\vec{p}_i(t) \cdot \delta \dot{\vec{q}}_i(t) - \frac{\partial H}{\partial \vec{q}_i} \cdot \delta \vec{q}_i
\right]\ dt
$$
Integrating the first term in the integral by parts gives:
$$
\delta J_{\vec{q}_n} = \frac{\partial L}{\partial \vec{q}_n(t_1)} \cdot \delta \vec{q}_n(t_1)  +
\left[
\vec{p}_i \cdot \delta \vec{q}_i
\right]_{t_0}^{t_1}
- \int_{t_0}^{t_1} \left[
\dot{\vec{p}}_i + \frac{\partial H}{\partial \vec{q}_i}
\right] \cdot \delta \vec{q}_i\ dt
$$
To make this variation zero for any $\delta\vec{q}_n$, we define $\vec{p}_n$ with the following dynamics (the **co-state equations**):
$$
\dot{\vec{p}}_n(t) = - \frac{\partial H}{\partial \vec{q}_n}
$$
And the terminal boundary condition (since $\delta\vec{q}_n(t_0)=0$):
$$
\vec{p}_n(t_1) = - \frac{\partial L}{\partial \vec{q}_n(t_1)}
$$

##### **Variation with respect to $\vec{p}_n$**
This variation must also be zero, which recovers our original constraint.
$$
\delta J_{\vec{p}_n} =
\int_{t_0}^{t_1} \delta \vec{p}_i \cdot \left[
\dot{\vec{q}}_i - \frac{\partial H}{\partial \vec{p}_i}
\right]\ dt
$$
This implies that the **state equations** are:
$$
\dot{\vec{q}}_n = \frac{\partial H}{\partial \vec{p}_n}
$$
(Note that $\frac{\partial H}{\partial \vec{p}_n} = f_n$, so this is indeed our original constraint).

##### **Variation with respect to $\theta$**
Finally, the variation with respect to the parameters $\theta$ is:
$$
\delta J_{\theta} = - \int_{t_0}^{t_1}
\frac{\partial H}{\partial \theta}\ \delta \theta
\ dt
$$
Since all other variations are zero, we have $\delta T_L = \delta J = \delta J_{\theta}$. This gives us the final gradient:
$$
\frac{d T_L}{d\theta} = - \int_{t_0}^{t_1}
\frac{\partial H}{\partial \theta}\ dt
$$
This Hamiltonian formulation provides a clear and systematic procedure for deriving parameter gradients in complex continuous-time models.