import numpy as np


def random_ddm(drift, threshold, ndt, rel_sp=.5, noise_constant=1, dt=0.001, max_rt=10):
    """Simulates behavior (rt and accuracy) according to the diffusion decision model.
    In this parametrization, it is assumed that 0 is the lower threshold,
    and, when rel_sp=1/2, the diffusion process starts halfway through the threshold value.
    Note
    ----
    This function is mainly for the posterior predictive calculations.
    It assumes that drift, threshold and ndt are provided as numpy.ndarray
    of shape (n_samples, n_trials).
    However, it also works when the rel_sp is given as a float.
    Drift, threshold and ndt should have the same shape.
    Parameters
    ----------
    drift : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Drift-rate of the diffusion decision model.
    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold of the diffusion decision model.
    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time of the diffusion decision model, in seconds.
    Other Parameters
    ----------------
    rel_sp : numpy.ndarray or float, default .5
        When is an array , shape is usually (n_samples, n_trials).
        Relative starting point of the diffusion decision model.
    noise_constant : float, default 1
        Scaling factor of the diffusion decision model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.
    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.
    dt : float, default 0.001
        Controls the time resolution of the diffusion decision model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.
    Returns
    -------
    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.
    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the diffusion decision model.
        Every element corresponds to the set of parameters given as input with the same shape.
    """

    shape = drift.shape

    acc = np.empty(shape)
    acc[:] = np.nan
    rt = np.empty(shape)
    rt[:] = np.nan
    rel_sp = np.ones(shape)*rel_sp
    max_tsteps = max_rt/dt

    # initialize the diffusion process
    x = np.ones(shape)*rel_sp*threshold
    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    # start accumulation process
    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x[ongoing] += np.random.normal(drift[ongoing]*dt,
                                       noise_constant*np.sqrt(dt),
                                       np.sum(ongoing))
        tstep += 1

        # ended trials
        ended_correct = (x >= threshold)
        ended_incorrect = (x <= 0)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt*tstep + ndt[np.logical_and(ended_correct,
                                                                                       ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt*tstep + ndt[np.logical_and(ended_incorrect,
                                                                                         ongoing)]
            ongoing[ended_incorrect] = False

    return rt, acc