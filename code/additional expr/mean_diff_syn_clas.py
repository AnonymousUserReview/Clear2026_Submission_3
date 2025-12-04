#%%
import numpy as np
import pandas as pd
from sklearn import svm
from tqdm.notebook import tqdm
from mechanism_learn import pipeline as mlpipe
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def mean_diff(X, Y):
    X1_mean_y1 = np.mean(X[Y.ravel()==-1], axis=0)
    X1_mean_y2 = np.mean(X[Y.ravel()==1], axis=0)
    expect_diff = X1_mean_y2 - X1_mean_y1
    return expect_diff


#%% Read synthetic data
data_path = r"../../test_data/synthetic_data/syn_classification/"

X_train_conf = pd.read_csv(data_path + "X_train_conf.csv").to_numpy()
Y_train_conf = pd.read_csv(data_path + "Y_train_conf.csv").to_numpy().ravel()
Z_train_conf = pd.read_csv(data_path + "Z_train_conf.csv").to_numpy()

X_train_unconf = pd.read_csv(data_path + "X_train_unconf.csv").to_numpy()
Y_train_unconf = pd.read_csv(data_path + "Y_train_unconf.csv").to_numpy().ravel()

X_test_unconf = pd.read_csv(data_path + "X_test_unconf.csv").to_numpy()
Y_test_unconf = pd.read_csv(data_path + "Y_test_unconf.csv").to_numpy().ravel()

X_test_conf = pd.read_csv(data_path + "X_test_conf.csv").to_numpy()
Y_test_conf = pd.read_csv(data_path + "Y_test_conf.csv").to_numpy().ravel()
#%% Parameters
# Parameter for resampling
n_samples = [ (Y_train_conf == c).sum() for c in np.unique(Y_train_conf)]
# Parameters for CWGMM
comp_k = 4
cov_type = 'full'
max_iter = 500
tol = 1e-6
init_method = 'kmeans++'
cov_reg = 1e-5
min_variance_value=1e-5
# Parameters for weights estimation
est_method = "histogram"
n_bins = [0, 20]
# Paramters for comparison and plotting
sample_size = [50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000]
save_fig = True
save_dir = r"../../res_fig/"
legend_dir = save_dir + r"legends/"
main_fig_name = "syn_clas_mean_diff.png"
legend_fig_name = "syn_clas_mean_diff_legend.png"
figsize = (6.5, 4)
labels = ["Confounded data", 
          "Unconfounded data", 
          "deconfounded data by CW-GMM", 
          "deconfounded data by CB", 
          "Theoretical (unconfounded)",
          "Theoretical (confounded)"]
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
                                                    intv_values = np.unique(Y_train_conf), 
                                                    est_method = est_method, 
                                                    n_bins = n_bins
                                                    )
ml_gmm_pipeline.cwgmm_fit(comp_k = comp_k,
                            max_iter = max_iter,
                            tol = tol,
                            init_method = init_method,
                            cov_type = cov_type,
                            cov_reg = cov_reg,
                            min_variance_value = min_variance_value,
                            return_model = False,
                            verbose = 0)
X_deconf_gmm, Y_deconf_gmm = ml_gmm_pipeline.cwgmm_resample(n_samples = n_samples,
                                                            return_samples = True)

ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                    mechanism_data = Z_train_conf, 
                                                    effect_data = X_train_conf, 
                                                    intv_values = np.unique(Y_train_conf), 
                                                    est_method = est_method, 
                                                    n_bins = n_bins)
X_deconf_cb, Y_deconf_cb = ml_cb_pipeline.cb_resample(n_samples = n_samples,
                                                      return_samples = True,
                                                      verbose = 0)

#%% Compute mean differences
mean_X1_diffs = {"Confounded data": [], "Unconfounded data": [], "deconfounded data by CW-GMM": [], "deconfounded data by CB": []}
mean_X2_diffs = {"Confounded data": [], "Unconfounded data": [], "deconfounded data by CW-GMM": [], "deconfounded data by CB": []}

for size in sample_size:
    mean_X1_diffs["Confounded data"].append(mean_diff(X_train_conf[:size, 0], Y_train_conf[:size]))
    mean_X1_diffs["Unconfounded data"].append(mean_diff(X_train_unconf[:size, 0], Y_train_unconf[:size]))
    mean_X1_diffs["deconfounded data by CW-GMM"].append(mean_diff(X_deconf_gmm[:size, 0], Y_deconf_gmm[:size]))
    mean_X1_diffs["deconfounded data by CB"].append(mean_diff(X_deconf_cb[:size, 0], Y_deconf_cb[:size]))
    
    mean_X2_diffs["Confounded data"].append(mean_diff(X_train_conf[:size, 1], Y_train_conf[:size]))
    mean_X2_diffs["Unconfounded data"].append(mean_diff(X_train_unconf[:size, 1], Y_train_unconf[:size]))
    mean_X2_diffs["deconfounded data by CW-GMM"].append(mean_diff(X_deconf_gmm[:size, 1], Y_deconf_gmm[:size]))
    mean_X2_diffs["deconfounded data by CB"].append(mean_diff(X_deconf_cb[:size, 1], Y_deconf_cb[:size]))

theoretical_ATE_X1 = 2.027
theoretical_ATE_X2 = 0.0
theoretical_confounded_effect_X1 = 2.64
theoretical_confounded_effect_X2 = 1.92

# %%

fig, ax = plt.subplots(1, 1, figsize=figsize)
for key, diffs in mean_X1_diffs.items():
    ax.plot(sample_size, diffs, marker=marker_style, label=key, linewidth = effect_lw, color= color_list[labels.index(key)] if key in labels[:-2] else None)
ax.plot(sample_size, [theoretical_ATE_X1]*len(sample_size), line_style, color = color_list[-2],
        label='theoretical ATE X1', linewidth = theoretical_lw)
ax.plot(sample_size, [theoretical_confounded_effect_X1]*len(sample_size), line_style, color = color_list[-1], 
        label='theoretical confounded effect X1', linewidth = theoretical_lw)
ax.set_xscale('log')
ax.set_xlabel('Sample Size (log scale)', fontsize=axis_label_fontsize)    
ax.set_ylabel('Expectation diff. of X1 between classes', fontsize=axis_label_fontsize)
ax.set_title('(a)', loc='left', fontsize=title_size)
ax.set_xlim(xlim[0], xlim[1])
ax.grid(True)
plt.tight_layout()
if save_fig:
    plt.savefig( "fdr_clas_ATE_X1.png", dpi=600)
plt.show()

handles = []
fig, ax = plt.subplots(1, 1, figsize=figsize)
for key, diffs in mean_X2_diffs.items():
    handle_i = ax.plot(sample_size, diffs, marker=marker_style, label=key, linewidth = effect_lw, color= color_list[labels.index(key)] if key in labels[:-2] else None)
    handles.append(handle_i[0])
handle_true = ax.plot(sample_size, [theoretical_ATE_X2]*len(sample_size), line_style, color = color_list[-2],
                      label='theoretical ATE X2', linewidth = theoretical_lw)
handle_conf = ax.plot(sample_size, [theoretical_confounded_effect_X2]*len(sample_size), line_style, color = color_list[-1],
                      label='theoretical confounded effect X2', linewidth = theoretical_lw)
handles.append(handle_true[0])
handles.append(handle_conf[0])
ax.set_xscale('log')
ax.set_xlabel('Sample Size (log scale)', fontsize=axis_label_fontsize)    
ax.set_ylabel('Expectation diff. of X2 between classes', fontsize=axis_label_fontsize)
ax.set_title('(b)', loc='left', fontsize=title_size)
ax.grid(True)
ax.set_xlim(xlim[0], xlim[1])
plt.tight_layout()
if save_fig:
    plt.savefig(save_dir + main_fig_name, dpi=600)
plt.show()

# save the legend
plt.figure(figsize=(21,1))
plt.axis('off')
legend =plt.legend(handles=handles,
                     labels=labels,
                     loc='center',
                     ncol=len(labels),
                     markerscale=2)
if save_fig:
    plt.savefig(legend_dir + legend_fig_name, dpi=300)
# %%
