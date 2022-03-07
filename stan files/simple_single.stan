data {
    int N;                                                  // Number of training samples
    vector[2] x[N];                                         // Feature data
    real rt[N];                                             // rt
    int<lower=-1,upper=1> accuracy[N];                      // accuracy (-1, 1)

    vector[2] drift_scaling_priors;                         // mean and sd of the prior
    vector[2] threshold_priors;                             // mean and sd of the prior
    vector[2] ndt_priors;                                   // mean and sd of the prior
    real starting_point;
}

parameters {
    real drift_scaling;
    real threshold;
    real ndt;
}
transformed parameters {
    real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
    real drift_t[N];								// trial-by-trial drift rate for predictions
    real<lower=0> threshold_t[N];                   // trial-by-trial threshold
    real<lower=0> ndt_t[N];                         // trial-by-trial ndt


    real transf_drift_scaling;
    real transf_threshold;
    real transf_ndt;

    transf_drift_scaling = log(1 + exp(drift_scaling));
    transf_threshold = log(1 + exp(threshold));
    transf_ndt = log(1 + exp(ndt));

    for (n in 1:N) {
            drift_t[n] = transf_drift_scaling * (log(x[n][1]/x[n][2]));
            //drift_t[n] = log(x[n][1]/x[n][2]);
            drift_ll[n] = drift_t[n]*accuracy[n];
            threshold_t[n] = transf_threshold;
            ndt_t[n] = transf_ndt;

    }
 }
model {
    drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
    threshold ~ normal(threshold_priors[1], threshold_priors[2]);
    ndt ~ normal(ndt_priors[1], ndt_priors[2]);
    
    rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}

// generated quantities {
// 	vector[N] log_lik;

// 	{for (n in 1:N) {
// 		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
// 	}
// 	}
// }