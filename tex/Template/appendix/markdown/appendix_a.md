### **Appendix A: Derivation of the Adjoint Method for Neural ODEs**

**Summary:**
*This appendix provides a detailed derivation of the adjoint method as presented in Appendix B of the paper [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366). The derivation here is presented in a more intuitive manner, aiming to deduce the result from first principles rather than starting with the definition of the adjoint state.*

#### **A.1 Context**

In a standard Recurrent Neural Network (RNN), a sequence of hidden states is generated, where each state carries information to the next:

$$
\vec{h}_{t+1} = \vec{h}_t + f(\vec{h}_t, \theta_t)
$$

Optimization in such models involves minimizing a loss function with respect to a set of parameters $\theta$. The Neural Ordinary Differential Equation (NODE) model extends this concept to the continuous-time domain. Instead of discrete steps, we consider infinitesimal time steps. As the time steps approach zero, the sequence of hidden states can be modeled as a continuous trajectory, $\vec{z}(t)$, governed by an ordinary differential equation (ODE):

$$
\frac{d\vec{z}(t)}{dt} = \dot{\vec{z}} = f(\vec{z}(t), \theta, t)
$$

We wish to find the gradient of a loss function $L$ with respect to the parameters $\theta$. The dependence of $L$ on $\theta$ is implicit; in the context of the paper, $L$ depends directly only on the state at the final time, $L(\vec{z}(t_1))$. To find this gradient, we use the method of Lagrange multipliers to construct a new functional, $\mathcal{J}$, which incorporates the dynamic constraint of the ODE.

$$
\mathcal{J}(\vec{z}, \vec{\lambda}, \theta) =
L(\vec{z}(t_1)) +
\int_{t_0}^{t_1} \vec{\lambda}^{T}(t)
\left[
f(\vec{z}(t), \theta, t) - \dot{\vec{z}}(t)
\right]
\ dt
$$

**Note:** By construction, the value of the functional $\mathcal{J}$ is identical to that of $L$ whenever the dynamic constraint is satisfied, as the integral term becomes zero.

Our goal is to find the total derivative of $L$ with respect to $\theta$. Since $\mathcal{J} = L$ when the constraint is satisfied, we can analyze the first variation of $\mathcal{J}$, denoted $\delta\mathcal{J}$. A small variation in the parameter $\theta$ induces a total variation in the functional, which can be expressed as:

$$
\delta \mathcal{J} = \delta L = \frac{d L}{d \theta}\ \delta \theta
$$

We calculate $\delta\mathcal{J}$ by considering the contributions from each of its arguments: $\vec{z}$, $\vec{\lambda}$, and $\theta$.

$$
\delta \mathcal{J} =
\frac{\delta \mathcal{J}}{\delta \vec{z}}\ \delta \vec{z} +
\frac{\delta \mathcal{J}}{\delta \theta}\ \delta \theta +
\frac{\delta \mathcal{J}}{\delta \vec{\lambda}}\ \delta \vec{\lambda}
$$

#### **A.2 First Variation of the Functional**

##### **Variation with Respect to $\vec{\lambda}$**
The variation of the functional with respect to the Lagrange multiplier $\vec{\lambda}$ yields:
$$
\frac{\delta \mathcal{J}}{\delta \vec{\lambda}} =
f(\vec{z}, \theta, t) - \dot{\vec{z}}
$$
The optimality condition requires that the coefficient of each independent variation be zero. Setting $\frac{\delta \mathcal{J}}{\delta \vec{\lambda}} = 0$ **recovers** the original dynamic constraint:
$$
\dot{\vec{z}} = f(\vec{z}, \theta, t)
$$

##### **Variation with Respect to $\theta$**
The contribution to the total variation from the explicit dependence on the parameter $\theta$ is:
$$
\frac{\delta \mathcal{J}}{\delta \theta}\ \delta \theta =
\int_{t_0}^{t_1} \vec{\lambda}^{T}(t)\
    \frac{\partial f}{\partial \theta}\ \delta \theta\ dt
$$

##### **Variation with Respect to $\vec{z}$**
The variation of $\mathcal{J}$ with respect to the trajectory $\vec{z}(t)$ is given by:
$$
\frac{\delta \mathcal{J}}{\delta \vec{z}}\ \delta \vec{z} =
\frac{\partial L}{\partial \vec{z}(t_1)}\ \delta \vec{z}(t_1) +
\int_{t_0}^{t_1} \vec{\lambda}^{T}(t)\left[
    \frac{\partial f}{\partial \vec{z}}
    \ \delta \vec{z} - \delta \dot{\vec{z}}\
\right]\ dt
$$
To handle the inconvenient $\delta \dot{\vec{z}}$ term, we apply integration by parts to $-\int_{t_0}^{t_1} \vec{\lambda}^T \delta \dot{\vec{z}} \ dt$:
$$
-\int_{t_0}^{t_1} \vec{\lambda}^T \delta \dot{\vec{z}} \ dt =
\int_{t_0}^{t_1} \dot{\vec{\lambda}}^T \delta\vec{z} \ dt -
\left[\ \vec{\lambda}^T \delta\vec{z}\ \right]_{t_0}^{t_1}
$$
Substituting this back and rearranging terms, we get:
$$
\frac{\delta \mathcal{J}}{\delta \vec{z}}\ \delta \vec{z} =
\left(
    \frac{\partial L}{\partial \vec{z}(t_1)} - \vec{\lambda}^T(t_1)
\right) \delta\vec{z}(t_1) +
\vec{\lambda}^T(t_0)\ \delta\vec{z}(t_0) +
\int_{t_0}^{t_1} \left[
    \vec{\lambda}^{T}\ \frac{\partial f}{\partial \vec{z}} +
    \dot{\vec{\lambda}}^{T}\
\right]\delta \vec{z} \ dt
$$
The term $\delta\vec{z}(t_0)$ is zero because the initial condition $\vec{z}(t_0)$ is fixed, and thus its variation is null. To nullify the remaining terms, which involve arbitrary variations ($\delta\vec{z}(t_1)$ and $\delta\vec{z}(t)$), we strategically define our **adjoint state** $\vec{\lambda}(t)$.

By imposing that the coefficient of each independent variation is zero, we obtain the following conditions:

1.  **Boundary Condition at Final Time ($t_1$):** To eliminate the boundary term at $t_1$, we set its coefficient to zero:
    $$
    \frac{\partial L}{\partial \vec{z}(t_1)} - \vec{\lambda}^T(t_1) = 0 \quad \implies \quad
    \vec{\lambda}^T(t_1) = \frac{\partial L}{\partial \vec{z}(t_1)}
    $$
    This defines the value of the adjoint state at the final time, which serves as the "initial condition" for its reverse-time integration.

2.  **Differential Equation of the Adjoint State:** To nullify the integral term for any variation $\delta\vec{z}(t)$, the integrand must be identically zero:
    $$
    \dot{\vec{\lambda}}^{T} + \vec{\lambda}^{T}\ \frac{\partial f}{\partial \vec{z}} = 0
    \quad \implies \quad
    \dot{\vec{\lambda}}^{T} = - \vec{\lambda}^{T}\ \frac{\partial f}{\partial \vec{z}}
    $$
    This is the differential equation that governs the dynamics of the adjoint state. By imposing these two conditions, we ensure that the entire variation $\frac{\delta \mathcal{J}}{\delta \vec{z}}$ vanishes.

#### **A.3 Final Result**

With the variations with respect to $\vec{z}$ and $\vec{\lambda}$ being zero (by construction and recovery of constraints), the total variation $\delta \mathcal{J}$ (which equals $\delta L$) simplifies to only the explicit contribution from $\theta$:

$$
\delta L = \frac{d L}{d \theta}\ \delta \theta =
\int_{t_0}^{t_1} \vec{\lambda}^{T}(t)\
    \frac{\partial f}{\partial \theta}\ \delta \theta\ dt
$$

Since the variation $\delta\theta$ is arbitrary, we can equate the coefficients to obtain the final expression for the gradient of the loss function:

$$
\frac{d L}{d \theta} =
\int_{t_0}^{t_1} \vec{\lambda}^{T}(t)\
    \frac{\partial f}{\partial \theta}\ dt
$$