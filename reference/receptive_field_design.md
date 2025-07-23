<font face='times new roman'>

# Receptive field theory
1. How to pick the time-meaned firing rate;
2. How to put the stimulus;
3. How to analyze the receptive field.

## Pick time-meaned firing rate
Use `mydata` and `firing_rate_analysis` to get $\nu(\vec{r},t)$, `len(t)`. Then sum.

## Stimulus type
1. Central point(s)
2. Gaussian field
3. Local uniform field

## External caused firing rate analysis
Repeat n realizations, get external caused firing rate $\nu_{ext}(\vec{r})$:

$$
\nu_{ext}(\vec{r}) = \nu_{sti}(\vec{r}) - \nu_{free}(\vec{r})
$$

In which, $\nu_{sti}(\vec{r})$ refers firing rate distribution with sitmulus, while $\nu_{free}(\vec{r})$ does not. Each $\nu(\vec{r})$: 

$$
\nu(\vec{r}) = \lim_{T \to \infty} \frac{1}{T}\int_T \nu(\vec{r},t)dt
$$

Numerical receptive field is easy to find, just make sure the graph line smooth enough and pick the numerical zero point, it's the radius of the receptive field.

For canonical analysis, due to the fitness, apply the following fitting:
1. If $\nu_{ext}$ with a Gaussian form:

    $$
    \nu_{ext} = A_+ e^{-r^2/2\sigma_+^2} - A_- e^{-r^2/2\sigma_-^2}
    $$

    Receptive field ratius:
<!-- 
    $$
    A_+ e^{-r^2/2\sigma_+^2} = A_- e^{-r^2/2\sigma_-^2}
    $$

    $$
    A_+/A_- = e^{r^2/2\sigma_+^2-r^2/2\sigma_-^2}
    $$

    $$
    2\ln(A_+)-2\ln(A_-)=r^2(1/\sigma_+^2-1/\sigma_-^2)
    $$ -->

    $$
    r = \sqrt{2\frac{\ln(A_+)-2\ln(A_-)}{(1/\sigma_+^2-1/\sigma_-^2)}}
    $$

2. If $\nu_{ext}$ with a exponential form:

    $$
    \nu_{ext} = A_+ e^{-r/\sigma_+} - A_- e^{-r/\sigma_-}
    $$

    Receptive field ratius:
    
    $$
    r = \frac{\ln(A_+)-2\ln(A_-)}{(1/\sigma_+-1/\sigma_-)}
    $$



