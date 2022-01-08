#!/opt/anaconda3/bin/python

import pickle
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pystan

infile = open("../check_points/all_feature_vec",'rb')
all_feature_vec = pickle.load(infile)
infile.close()

data = []
acc = []
rt = []
correct_option = []
drift = []
w_prob = []
nonw_prob = []
vocab = []
with open("../Datasets/train_dataset_rt_acc.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:

        vocab.append(row[0])
        data.append(row)
        acc.append(row[5]) # acc ouput of DDM model
        rt.append(float(row[2]))
        drift.append(row[6])
        w_prob.append(row[4])
        nonw_prob.append(row[3])
        if row[1] == "nonword":
            correct_option.append(1)
        else:
            correct_option.append(2)



model = """
functions {
		vector[] nn_predict(matrix x, vector d_t_h1, vector d_t_h2,vector y_bias) {
			int N = rows(x);
			int num_labels = cols(y_bias) + 1;
			vector[num_labels] output_layer_logit[N];
			vector[N] ones = rep_vector(1., N);
			real b1 = y_bias[1];
			real b2 = y_bias[2];
			output_layer_logit[1:N, 1] = to_array_1d(x * d_t_h1 + b1);
			output_layer_logit[1:N, 2] = to_array_1d(x * d_t_h2 + b2);
			
			return(output_layer_logit);
		}
}

data {
		int N; // Number of training samples
		int P; // Number of predictors (features)
		matrix[N, P] x; // Feature data
		int labels[N]; // Outcome labels
		int K;									// number of options
		real rt[N];							// rt
		//real starting_point;			// starting point diffusion model not to estimate
		vector[2] drift_scaling_priors;					// mean and sd of the prior
		vector[2] threshold_priors;						// mean and sd of the prior
		vector[2] ndt_priors;							// mean and sd of the prior
}

parameters {

		vector[P] alpha_data_to_hidden_weights1; // Data -> Hidden 1
		vector[P] alpha_data_to_hidden_weights2; // Data -> Hidden 1
		vector[K] alpha_y_bias; // Bias. 

		real drift_scaling;
		real threshold;
		real ndt;
}
transformed parameters {
		real drift[N];								// trial-by-trial drift rate for predictions
		real threshold_t[N];					// trial-by-trial threshold
		real ndt_t[N];							// trial-by-trial ndt

		vector[K] Q_output_layer_logit[N];						// Q state values

		real transf_drift_scaling;
		real transf_threshold;
		real transf_ndt;
		
		Q_output_layer_logit = nn_predict(x,
																		alpha_data_to_hidden_weights1,
																		alpha_data_to_hidden_weights2,
																		alpha_y_bias);

		transf_drift_scaling = log(1 + exp(drift_scaling));
		transf_threshold = log(1 + exp(threshold));
		transf_ndt = log(1 + exp(ndt));
		for (n in 1:N) {
			drift[n] = transf_drift_scaling * log(inv_logit(Q_output_layer_logit[n][2])/inv_logit(Q_output_layer_logit[n][1]));
			threshold_t[n] = transf_threshold;
			ndt_t[n] = transf_ndt;
			
			}
 }
model {
		to_vector(alpha_data_to_hidden_weights1) ~ std_normal();
		to_vector(alpha_data_to_hidden_weights2) ~ std_normal();

		alpha_y_bias ~ std_normal();
		drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
		threshold ~ normal(threshold_priors[1], threshold_priors[2]);
		ndt ~ normal(ndt_priors[1], ndt_priors[2]);

		rt ~ wiener(threshold_t, ndt_t, transf_threshold/2, drift);

		for(n in 1:N) { // Likelihood
			labels[n] ~ categorical_logit(Q_output_layer_logit[n]);
		}

}



"""


data = {
          'N' :  len(all_feature_vec) ,# Number of training samples
          'P': len(all_feature_vec[0]),# Number of predictors (features)
          'x' : all_feature_vec, # Feature data
          'labels' : correct_option,  # Outcome
          'K': 2,									# number of options
          'rt': rt,
          # 'starting_point': .5,
        	'drift_scaling_priors' : [0, 1],					#mean and sd of the prior
          'threshold_priors' : [5, 1],						# mean and sd of the prior
          'ndt_priors':[0, 1] 	
        
}


sm = pystan.StanModel(model_code=model, extra_compile_args=["-w"], verbose=True)


# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, seed=1, verbose=True)

print(pystan.check_hmc_diagnostics(fit))

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])
print(df)

for f in df["Rhat"]:
    if f > 1.01:
        print(f)


outfile = open("check_points/stan_fit",'wb')
pickle.dump(fit,outfile)
outfile.close()

outfile = open("check_points/stan_model",'wb')
pickle.dump(sm,outfile)
outfile.close()

df.to_pickle("check_points/summary.pkl")
