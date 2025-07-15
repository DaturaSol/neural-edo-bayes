### **Appendix C: Derivation of KL Divergence for Univariate Gaussians**

#### **C.1 The Definition**

The Kullback-Leibler (KL) divergence between two continuous probability distributions, $P$ and $Q$, is defined by the integral:
$$
D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{p(x)}{q(x)}\right) dx
$$
Let $P$ be the "true" distribution $\mathcal{N}(\mu_0, \sigma_0^2)$ and $Q$ be the "approximating" distribution $\mathcal{N}(\mu_1, \sigma_1^2)$. The formula can be rewritten as an expectation with respect to the distribution $P$:
$$
D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log\left(\frac{p(x)}{q(x)}\right) \right]
$$
Using the properties of logarithms, we can split this into two terms:
$$
D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log(p(x)) \right] - \mathbb{E}_{x \sim P} \left[ \log(q(x)) \right]
$$
The first term is the negative **differential entropy** of P, and the second term is the **cross-entropy** between P and Q.

#### **C.2 Step 1: Solving the Cross-Entropy Term**

We first evaluate the term $\mathbb{E}_{x \sim P} \left[ \log(q(x)) \right]$. The probability density function (PDF) for $q(x)$ is:
$$
q(x) = \frac{1}{\sqrt{2\pi\sigma_1^2}} \exp\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)
$$
The natural logarithm of the PDF is:
$$
\log(q(x)) = -\log(\sqrt{2\pi\sigma_1^2}) - \frac{(x-\mu_1)^2}{2\sigma_1^2} = -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{(x-\mu_1)^2}{2\sigma_1^2}
$$
Now, we take the expectation of this expression with respect to $x \sim P$.
$$
\mathbb{E}_{P}\left[ \log(q(x)) \right] = \mathbb{E}_{P}\left[ -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{(x-\mu_1)^2}{2\sigma_1^2} \right]
$$
Using the linearity of expectation:
$$
= -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{1}{2\sigma_1^2} \mathbb{E}_{P}\left[ (x-\mu_1)^2 \right]
$$
To evaluate $\mathbb{E}_{P}\left[ (x-\mu_1)^2 \right]$, we use the fact that for $x \sim P$, $\mathbb{E}[x] = \mu_0$ and $\mathbb{E}[(x - \mu_0)^2] = \sigma_0^2$.
$$
\mathbb{E}_{P}\left[ (x-\mu_1)^2 \right] = \mathbb{E}_{P}\left[ ((x-\mu_0) + (\mu_0-\mu_1))^2 \right]
$$
$$
= \mathbb{E}_{P}\left[ (x-\mu_0)^2 + 2(x-\mu_0)(\mu_0-\mu_1) + (\mu_0-\mu_1)^2 \right]
$$
$$
= \mathbb{E}_{P}[(x-\mu_0)^2] + 2(\mu_0-\mu_1)\mathbb{E}_{P}[x-\mu_0] + (\mu_0-\mu_1)^2
$$
Since $\mathbb{E}_{P}[x-\mu_0] = \mu_0 - \mu_0 = 0$, the middle term vanishes. We are left with:
$$
\mathbb{E}_{P}\left[ (x-\mu_1)^2 \right] = \sigma_0^2 + (\mu_0-\mu_1)^2
$$
Plugging this back into our expression for the cross-entropy:
$$
\mathbb{E}_{P}\left[ \log(q(x)) \right] = -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2}
$$

#### **C.3 Step 2: The Entropy Term**

The first term, $\mathbb{E}_{P}\left[ \log(p(x)) \right]$, is the negative of the differential entropy of a Gaussian distribution. This is a standard result, which can be derived similarly to the step above. The value is:
$$
\mathbb{E}_{P}\left[ \log(p(x)) \right] = -\frac{1}{2}\log(2\pi\sigma_0^2) - \frac{1}{2} = -\frac{1}{2}\log(2\pi) - \log(\sigma_0) - \frac{1}{2}
$$

#### **C.4 Step 3: Assembling the Final Result**

Finally, we compute $D_{KL} = \mathbb{E}_{P}[\log(p(x))] - \mathbb{E}_{P}[\log(q(x))]$:
$$
D_{KL} = \left( -\frac{1}{2}\log(2\pi) - \log(\sigma_0) - \frac{1}{2} \right) - \left( -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2} \right)
$$
The $-\frac{1}{2}\log(2\pi)$ terms cancel. We are left with:
$$
= - \log(\sigma_0) - \frac{1}{2} + \log(\sigma_1) + \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2}
$$
Rearranging the terms to a more conventional form:
$$
= \log(\sigma_1) - \log(\sigma_0) + \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2} - \frac{1}{2}
$$
$$
D_{KL}(P \parallel Q) = \log\left(\frac{\sigma_1}{\sigma_0}\right) + \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2} - \frac{1}{2}
$$
