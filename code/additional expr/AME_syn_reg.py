#%%
import numpy as np
import pandas as pd
from sklearn import svm
from tqdm.notebook import tqdm
from mechanism_learn import pipeline as mlpipe
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import linregress
warnings.filterwarnings("ignore")


def ame(X, Y):
    slope, intercept, r_value, p_value, std_err = linregress(Y.ravel(), X.ravel())
    return slope

#%% Read synthetic data
data_path = r"../../test_data/synthetic_data/syn_regression/"
X_train_conf = pd.read_csv(data_path + "X_train_conf.csv").to_numpy()
Y_train_conf = pd.read_csv(data_path + "Y_train_conf.csv").to_numpy().ravel()
Z_train_conf = pd.read_csv(data_path + "Z_train_conf.csv").to_numpy()

X_train_unconf = pd.read_csv(data_path + "X_train_unconf.csv").to_numpy()
Y_train_unconf = pd.read_csv(data_path + "Y_train_unconf.csv").to_numpy().ravel()

X_test_conf = pd.read_csv(data_path + "X_test_conf.csv").to_numpy()
Y_test_conf = pd.read_csv(data_path + "Y_test_conf.csv").to_numpy().ravel()

X_test_unconf = pd.read_csv(data_path + "X_test_unconf.csv").to_numpy()
Y_test_unconf = pd.read_csv(data_path + "Y_test_unconf.csv").to_numpy().ravel()

#%% Parameters
# Parameters for CWGMM
comp_k = 2
max_iter = 1000
tol = 1e-4
cov_type = 'full'
cov_reg = 1e-4
min_variance_value = 1e-5
init_method = 'kmeans++'
# Parameters for weights estimation
est_method = "multinorm"
N = X_train_conf.shape[0]
intv_intval_num = 50
Y_interv_values = np.linspace(Y_train_conf.min()*1.6, Y_train_conf.max()*1.3, intv_intval_num)
n_samples = [int(N // intv_intval_num)] * intv_intval_num
# Paramters for comparison and plotting
sample_size = [50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000]
save_fig = True
save_dir = r"../../res_fig/"
legend_dir = save_dir + r"legends/"
main_fig_name = "syn_regs_AME.png"
legend_fig_name = "syn_regs_AME_legend.png"
figsize = (6.5, 4)
labels = ["Confounded data", 
          "Unconfounded data", 
          "deconfounded data by CW-GMM", 
          "deconfounded data by CB", 
          "Theoretical AME",
          "Confounded AME"]
color_list = ["cornflowerblue", "orange", "#5CAA66", "tomato", "black", "dimgray"]
effect_lw = 1.5
theoretical_lw = 2.5
line_style = "--"
marker_style = "o"
xlim = (40, 60000)
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] =16
plt.rcParams["ytick.labelsize"] =16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] =16
plt.rcParams["axes.titlesize"] = 16
title_size = 16
axis_label_fontsize = 14
#%% Deconfounding with CW-GMM and CB
ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                    mechanism_data = Z_train_conf, 
                                                    effect_data = X_train_conf, 
                                                    intv_values = Y_interv_values, 
                                                    dist_map = None, 
                                                    est_method = est_method
                                                    )

ml_gmm_pipeline.cwgmm_fit(comp_k = comp_k,
                          max_iter = max_iter,
                          tol = tol,
                          init_method= init_method,
                          cov_type = cov_type,
                          cov_reg = cov_reg, 
                          min_variance_value=min_variance_value, 
                          verbose = 1)

X_deconf_gmm, Y_deconf_gmm = ml_gmm_pipeline.cwgmm_resample(n_samples = n_samples, return_samples = True)

ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                   mechanism_data = Z_train_conf, 
                                                   effect_data = X_train_conf, 
                                                   intv_values = Y_interv_values, 
                                                   dist_map = None, 
                                                   est_method = est_method
                                                   )

X_deconf_cb, Y_deconf_cb = ml_cb_pipeline.cb_resample(n_samples = n_samples,
                                                      return_samples = True)

#%%
mean_X_diffs = {"Confounded data": [], "Unconfounded data": [], "deconfounded data by CW-GMM": [], "deconfounded data by CB": []}

for size in sample_size:
    mean_X_diffs["Confounded data"].append(ame(X_train_conf[:size], Y_train_conf[:size]))
    mean_X_diffs["Unconfounded data"].append(ame(X_train_unconf[:size], Y_train_unconf[:size]))
    mean_X_diffs["deconfounded data by CW-GMM"].append(ame(X_deconf_gmm[:size], Y_deconf_gmm[:size]))
    mean_X_diffs["deconfounded data by CB"].append(ame(X_deconf_cb[:size], Y_deconf_cb[:size]))

theoretical_AME = 3.0
confounded_AME = 6.0

#%% Plotting

fig, ax = plt.subplots(1, 1, figsize=figsize)
handles = []
for key, diffs in mean_X_diffs.items():
    handle_i = ax.plot(sample_size, diffs, marker=marker_style, label=key, linewidth = effect_lw, color= color_list[labels.index(key)] if key in labels[:-2] else None)
    handles.append(handle_i[0])
handle_true = ax.plot(sample_size, [theoretical_AME]*len(sample_size), line_style, color = color_list[-2], label='theoretical unconfounded AME of Y on X', linewidth = theoretical_lw)
handle_conf = ax.plot(sample_size, [confounded_AME]*len(sample_size), line_style, color = color_list[-1], label='confounded theoretical AME of Y on X', linewidth = theoretical_lw)
handles.append(handle_true[0])
handles.append(handle_conf[0])
ax.set_xscale('log')
ax.set_xlabel('Sample Size (log scale)', fontsize=axis_label_fontsize)    
ax.set_ylabel('Average marginal effect of Y on X', fontsize=axis_label_fontsize)
ax.set_title('(c)', loc='left', fontsize=title_size)
ax.set_xlim(xlim[0], xlim[1])
ax.grid(True)
plt.tight_layout()
if save_fig:
    plt.savefig(save_dir + main_fig_name, dpi=600)
plt.show()
# %%
plt.figure(figsize=(4,2))
plt.axis('off')
legend =plt.legend(handles=handles,
                     labels=labels,
                     loc='center',
                     ncol=1,
                     markerscale=2)
if save_fig:
    plt.savefig(legend_dir + legend_fig_name, dpi=300)

# %%
