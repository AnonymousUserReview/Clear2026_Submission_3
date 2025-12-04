#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mechanism_learn import pipeline as mlpipe
import mechanism_learn.gmmSampler as gmms
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        # Jupyter notebook / JupyterLab
        from tqdm.notebook import tqdm
    else:
        # IPython Terminal, etc.
        from tqdm import tqdm
except (NameError, ImportError):
    # Standard Python interpreter
    from tqdm import tqdm

def visual_gaussian_comp(fig, ax, 
                         mus, sigmas, pi, 
                         AIC, BIC, 
                         X, y, x_range, y_range,
                         set_x_label, set_y_label, 
                         dot_color_set = None, contour_color_set = None,
                         cov_type = 'full',
                         n_levels = 20, min_level = 0.02, max_level = 0.95):
    
    intv_value = int(y.reshape(-1)[0])
    centroids_dot_clst = dot_color_set[::-1]
    K = len(mus)
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 200), np.linspace(y_range[0], y_range[1], 200))
    grid_X = np.c_[xx.ravel(), yy.ravel()]
    grid_probs = np.zeros((len(grid_X),K))
    comp_pdf = np.zeros_like(xx)
    
    for k in range(K):
        logp = gmms.log_gaussian_pdf(grid_X, mus[k], sigmas[k], cov_type=cov_type)
        grid_probs[:,k] = pi[k]*np.exp(logp)
        comp_pdf += grid_probs[:,k].reshape(xx.shape)
    levels = np.linspace(comp_pdf.max()*min_level, comp_pdf.max()*max_level, n_levels)
    cs = ax.contourf(xx, yy, comp_pdf, levels=levels, alpha=1, cmap=contour_color_set[int(intv_value/2+0.5)], extend="max")
    cbar = fig.colorbar(cs, ax=ax, label="Prob. density", format='%.3f')
    # Centroids of the components
    ax.scatter(X[:,0], X[:,1], s=dot_size, alpha=dot_alpha, c = dot_color_set[int(intv_value/2+0.5)], 
               label=r'$Y={}$ (class {})'.format(intv_value, intv_value))
    ax.scatter(mus[:,0], mus[:,1], s=500*pi, marker='X', c=centroids_dot_clst[int(intv_value/2+0.5)], 
               label=r"Class {} Comp. centroids (Size$\propto$Weight)".format(intv_value))

    if set_x_label:
        ax.set_xlabel('X1', fontsize=axis_label_fontsize)
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    if set_y_label:
        ax.set_ylabel('X2', fontsize=axis_label_fontsize)
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_title("K={} for interventional value y={} \n (AIC={:.2f}, BIC={:.2f})".format(mus.shape[0], int(y.reshape(-1)[0]), AIC, BIC))
    return ax
    
#%%
# Load synthetic confounded data
syn_data_dir = r"../../test_data/synthetic_data/"
testcase_dir = r"syn_classification/"
X_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_train_conf.csv").to_numpy()
Y_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv").to_numpy().ravel()
Z_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv").to_numpy()

# random sample 5000 from training set for faster training
idx_conf = np.random.choice(len(X_train_conf), 5000, replace=False)
X_train_conf = X_train_conf[idx_conf]
Y_train_conf = Y_train_conf[idx_conf]
Z_train_conf = Z_train_conf[idx_conf]

# Parameters for CWGMM
comp_k_lst = [2, 4, 6, 8, 10]
max_iter = 1000
tol = 1e-6
cov_type = 'full'
cov_reg = 1e-4
min_variance_value = 1e-2
init_method = "kmeans++"

# Parameters for weights estimation
est_method = "histogram"
n_bins = [0, 20]

# Parameters for resampling
n_sample = [1500, 1500]

# Parameters for visualization
save_fig = True
save_dir = r"../../res_fig/"
legends_dir = save_dir + r"legends/"
main_figure_name = "GMM_with_diff_K_horizontal.png"
legends_figure_name = "legend_syn_clas_compk.png"
figsize = (25, 9)
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.titlesize"] = 18
n_levels = 50
min_level_intv_1 = [0.05, 0.01, 0.01, 0.005, 0.005]
max_level_intv_1 = [0.8, 0.65, 0.6, 0.12, 0.45]
min_level_intv_2 = [0.05, 0.005, 0.001, 0.001, 0.001]
max_level_intv_2 = [0.85, 0.65, 0.1, 0.1, 0.1]
x_range = [-6, 6]
y_range = [-6, 6]
xaxis = [0, 0]
yaxis = [0, 0]
dot_color_set = ['royalblue', 'red']
contour_color_set = [plt.cm.Reds, plt.cm.Blues]
confounded_boundary_color = "orange"
true_boundary_color = "black"
dot_alpha = 0.3
dot_size = 10
axis_label_fontsize = 18
#%% Ploting GMMs with different K
results = {}  # key: comp_k, value: dict of results

pbar = tqdm(enumerate(comp_k_lst), total = len(comp_k_lst), desc="Fitting CWGMM with different K", leave=False, dynamic_ncols=True)
for i, comp_k in pbar:

    ml_gmm_pipeline = mlpipe.mechanism_learning_process(
        cause_data = Y_train_conf,
        mechanism_data = Z_train_conf, 
        effect_data = X_train_conf, 
        intv_values = np.unique(Y_train_conf), 
        dist_map = None, 
        est_method = est_method, 
        n_bins = n_bins
    )

    cwgmm_model = ml_gmm_pipeline.cwgmm_fit(
        comp_k = comp_k,
        max_iter = max_iter, 
        tol = tol, 
        init_method = init_method, 
        cov_type = cov_type, 
        cov_reg = cov_reg, 
        min_variance_value = min_variance_value, 
        random_seed = 42, 
        return_model = True,
        verbose = 1
    )
    
    deconf_X, deconf_Y = ml_gmm_pipeline.cwgmm_resample(
        n_samples = n_sample, 
        return_samples = True
    )
    
    gmm_params = cwgmm_model.params
    scores     = cwgmm_model.scores

    # intv1: Y=-1
    cwgmm_1_params = gmm_params[0]
    cwgmm_1_scores = scores[0]

    mus_est_intv1    = cwgmm_1_params['mus']
    Sigmas_est_intv1 = cwgmm_1_params['Sigmas']
    pi_est_intv1     = cwgmm_1_params['pi']
    AIC_intv1        = cwgmm_1_scores['AIC']
    BIC_intv1        = cwgmm_1_scores['BIC']

    deconf_samples_intv1 = deconf_X[deconf_Y[:,0] == -1]
    deconf_Y_intv1       = deconf_Y[deconf_Y[:,0] == -1]

    # intv2: Y=1
    cwgmm_2_params = gmm_params[1]
    cwgmm_2_scores = scores[1]

    mus_est_intv2    = cwgmm_2_params['mus']
    Sigmas_est_intv2 = cwgmm_2_params['Sigmas']
    pi_est_intv2     = cwgmm_2_params['pi']
    AIC_intv2        = cwgmm_2_scores['AIC']
    BIC_intv2        = cwgmm_2_scores['BIC']

    deconf_samples_intv2 = deconf_X[deconf_Y[:,0] ==  1]
    deconf_Y_intv2       = deconf_Y[deconf_Y[:,0] ==  1]

    # Save results for visualization
    results[comp_k] = dict(
        mus1    = mus_est_intv1,
        Sigmas1 = Sigmas_est_intv1,
        pi1     = pi_est_intv1,
        AIC1    = AIC_intv1,
        BIC1    = BIC_intv1,
        X1      = deconf_samples_intv1,
        Y1      = deconf_Y_intv1,

        mus2    = mus_est_intv2,
        Sigmas2 = Sigmas_est_intv2,
        pi2     = pi_est_intv2,
        AIC2    = AIC_intv2,
        BIC2    = BIC_intv2,
        X2      = deconf_samples_intv2,
        Y2      = deconf_Y_intv2,
    )
    
#%%
fig, axes = plt.subplots(2, len(comp_k_lst), figsize=figsize)
for i, comp_k in enumerate(comp_k_lst):
    res = results[comp_k]

    set_y_label = (i == 0)

    # ---- intv1: Y=-1 ----
    axes[0, i] = visual_gaussian_comp(
        fig, axes[0, i],
        res['mus1'], res['Sigmas1'], res['pi1'], res['AIC1'], res['BIC1'],
        res['X1'], res['Y1'], 
        cov_type=cov_type,
        dot_color_set=dot_color_set,
        contour_color_set=contour_color_set,
        x_range=x_range, y_range=y_range,
        set_x_label=False, set_y_label=set_y_label,
        n_levels=n_levels,
        min_level=min_level_intv_1[i],
        max_level=max_level_intv_1[i]
    )
    axes[0, i].plot(x_range, yaxis, color="orange", linewidth=3, label="Confounder boundary")
    axes[0, i].plot(xaxis, y_range, color="black", linewidth=3, label="True boundary")

    # ---- intv2: Y=1 ----
    axes[1, i] = visual_gaussian_comp(
        fig, axes[1, i],
        res['mus2'], res['Sigmas2'], res['pi2'], res['AIC2'], res['BIC2'],
        res['X2'], res['Y2'], 
        cov_type=cov_type,
        dot_color_set=dot_color_set,
        contour_color_set=contour_color_set,
        x_range=x_range, y_range=y_range,
        set_x_label=True, set_y_label=set_y_label,
        n_levels=n_levels,
        min_level=min_level_intv_2[i],
        max_level=max_level_intv_2[i]
    )
    axes[1, i].plot(x_range, yaxis, color=confounded_boundary_color, linewidth=3, label="Confounder boundary")
    axes[1, i].plot(xaxis, y_range, color=true_boundary_color, linewidth=3, label="True boundary")

plt.tight_layout()
if save_fig:
    plt.savefig(save_dir + main_figure_name, bbox_inches='tight', dpi=600)
plt.show()
#%%
fig_legend = plt.figure(figsize=(20, 1))
class1_legend = Line2D([], [], marker='o', color=dot_color_set[0], markersize=5, 
                       alpha=1, linestyle='None', label=r'$Y=-1$')
class2_legend = Line2D([], [], marker='o', color=dot_color_set[1], markersize=5, 
                       alpha=1, linestyle='None', label=r'$Y=1$')
centroids1_legend = Line2D([], [], marker='X', color=dot_color_set[1], markersize=10, 
                           alpha=1, linestyle='None', label=r"Class -1 Comp. mean (Size$\propto$Weight)")
centroids2_legend = Line2D([], [], marker='X', color=dot_color_set[0], markersize=10, 
                           alpha=1, linestyle='None', label=r"Class 1 Comp. mean (Size$\propto$Weight)")
true_boundary_legend = Line2D([], [], color=true_boundary_color, linewidth=3, label="True boundary")
confounder_boundary_legend = Line2D([], [], color=confounded_boundary_color, linewidth=3, label="Confounder boundary")
handles = [class1_legend, 
           class2_legend, 
           centroids1_legend, 
           centroids2_legend, 
           true_boundary_legend, 
           confounder_boundary_legend]
fig_legend.legend(handles = handles, loc='center', ncol=len(handles))
plt.axis('off') 
if save_fig:
    plt.savefig(legends_dir + legends_figure_name, bbox_inches='tight', dpi=300)
plt.show()
# axes[1].minorticks_on()
# %%
