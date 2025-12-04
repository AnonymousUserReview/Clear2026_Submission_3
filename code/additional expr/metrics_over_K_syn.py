#%%
from mechanism_learn import pipeline as mlpipe
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
import warnings
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
warnings.simplefilter('ignore')

# Plotting settings
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] =14
plt.rcParams["ytick.labelsize"] =14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] =16
plt.rcParams["axes.titlesize"] = 16

# Testing different component numbers
comp_k_lst = [2, 5, 10, 20, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
# Initialize or load the results dataframe
save_path = r"./results/"
filename = "metrics_container_simulations.csv"
try:
    metrics_container = pd.read_csv(save_path + filename, index_col=0)
except:
    metrics_container = pd.DataFrame(index=comp_k_lst, 
                                      columns=["Synthetic classification task",
                                               "Semi-synthetic classification task"])
    metrics_container.to_csv(save_path + filename, index=True)

def maxPooling_imgArr(img_flatArr, kernel_size, padding = "nearest", flatten = False):
    n_imgs = img_flatArr.shape[0]
    img_size = int(img_flatArr.shape[1]**0.5)
    img_arr = img_flatArr.reshape(n_imgs, img_size, img_size)
    resized_imgs = []
    for i in range(n_imgs):
        resized_imgs.append(maximum_filter(img_arr[i], size=kernel_size, mode=padding)[::kernel_size, ::kernel_size])
    resized_imgs = np.array(resized_imgs)
    if flatten:
        resized_imgs = resized_imgs.reshape(n_imgs, -1)
    return resized_imgs

def fmt_sci(x):
    """
    Format a number in scientific notation for matplotlib axis ticks.
    """
    s = f"{x:.1e}"       
    mantissa_str, exp_str = s.split("e")
    d = int(exp_str)
    return rf'${mantissa_str}\times10^{{{d}}}$'
#%%
syn_data_dir = r"../../test_data/synthetic_data/"
testcase_dir = r"syn_classification/"
X_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_train_conf.csv")
Y_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv")
Z_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv")
X_train_conf = np.array(X_train_conf)
Y_train_conf = np.array(Y_train_conf).reshape(-1,1)
Z_train_conf = np.array(Z_train_conf)

idx_conf = np.random.choice(len(X_train_conf), 5000, replace=False)
X_train_conf = X_train_conf[idx_conf]
Y_train_conf = Y_train_conf[idx_conf]
Z_train_conf = Z_train_conf[idx_conf]

X_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv")
Y_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv")
X_test_unconf = np.array(X_test_unconf)
Y_test_unconf = np.array(Y_test_unconf).reshape(-1,1)

# Sampling parameters
n_samples = [5000, 5000]
# CWGMM parameters
cov_type = 'full'
max_iter = 1000
tol = 1e-6
cov_reg = 1e-4
min_variance_value = 1e-4
init_method = 'kmeans++'
# Causal weights estimation method
est_method = "histogram"
n_bins = [0, 20]

pbar = tqdm(enumerate(comp_k_lst), total = len(comp_k_lst), desc="Exprs. running...", leave=False, dynamic_ncols=True)
for i, comp_k in pbar:
    ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                        mechanism_data = Z_train_conf, 
                                                        effect_data = X_train_conf, 
                                                        intv_values = np.unique(Y_train_conf), 
                                                        dist_map = None, 
                                                        est_method = est_method, 
                                                        n_bins = n_bins
                                                        )

    cwgmm_model = ml_gmm_pipeline.cwgmm_fit(comp_k = comp_k,
                                            max_iter = max_iter, 
                                            tol = tol, 
                                            init_method = init_method, 
                                            cov_type = cov_type, 
                                            cov_reg = cov_reg,
                                            min_variance_value=min_variance_value,
                                            random_seed=None, 
                                            return_model = True,
                                            verbose = 0)
    
    deconf_X_gmm, deconf_Y_gmm = ml_gmm_pipeline.cwgmm_resample(n_samples=n_samples, return_samples = True)


    deconf_gmm_clf = ml_gmm_pipeline.deconf_model_fit(ml_model = svm.SVC(kernel = 'linear', C=5))


    y_pred_gmm_deconf_unconf = deconf_gmm_clf.predict(X_test_unconf)
    
    acc = accuracy_score(Y_test_unconf, y_pred_gmm_deconf_unconf)
    pbar.set_postfix({"Component #": comp_k, "Accuracy": np.round(acc*100,2)})
    pbar.refresh()
    
    metrics_container.loc[comp_k, "Synthetic classification task"] = acc
    metrics_container.to_csv(save_path + "metrics_container_simulations.csv", index=True, header=True)
pbar.close()

#%%
semisyn_data_dir = r"./semi_synthetic_data/frontdoor_data/"

# CWGMM parameters
max_iter = 500
cov_reg = 1e-3
min_variance_value = 2e-3
tol = 1e-2
cov_type = "diag"
init_method = 'kmeans++'
# Sampling parameters
n_samples = [(Y_train_conf == 0).sum()*10, (Y_train_conf == 1).sum()*10]

X_train_conf = pd.read_csv(semisyn_data_dir + "X_train_conf.csv")
Y_train_conf = pd.read_csv(semisyn_data_dir + "Y_train_conf.csv")
Z_train_conf = pd.read_csv(semisyn_data_dir + "Z_train_conf.csv")
X_train_conf = np.array(X_train_conf)
X_train_conf = maxPooling_imgArr(X_train_conf, kernel_size=3, flatten=True)
Y_train_conf = np.array(Y_train_conf).reshape(-1,1)-1
Z_train_conf = np.array(Z_train_conf).reshape(-1,1)

X_test_unconf = pd.read_csv(semisyn_data_dir + "X_test_unconf.csv")
Y_test_unconf = pd.read_csv(semisyn_data_dir + "Y_test_unconf.csv")
X_test_unconf = np.array(X_test_unconf)
X_test_unconf = maxPooling_imgArr(X_test_unconf, kernel_size=3, flatten=True)
Y_test_unconf = np.array(Y_test_unconf).reshape(-1,1)-1

pbar = tqdm(enumerate(comp_k_lst), total = len(comp_k_lst), desc="Exprs. running...", leave=False, dynamic_ncols=True)
for i, comp_k in pbar:
    # print("Expr #{}/{}: comp_k = {}".format(i+1, len(comp_k_lst), comp_k))
    ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                        mechanism_data = Z_train_conf, 
                                                        effect_data = X_train_conf, 
                                                        intv_values = np.unique(Y_train_conf), 
                                                        est_method = "histogram", 
                                                        n_bins = [0, 0]
                                                        )
    ml_gmm_pipeline.cwgmm_fit(comp_k = comp_k,
                              max_iter = max_iter,
                              tol = tol,
                              cov_type = cov_type,
                              cov_reg = cov_reg,
                              min_variance_value = min_variance_value,
                              return_model = False,
                              verbose = 0)
    ml_gmm_pipeline.cwgmm_resample(n_samples = n_samples,
                                   return_samples = True)
    deconf_gmm_clf = ml_gmm_pipeline.deconf_model_fit(ml_model = KNeighborsClassifier(n_neighbors = 3))

    y_pred_gmm_deconf_unconf = deconf_gmm_clf.predict(X_test_unconf)
    
    acc = accuracy_score(Y_test_unconf, y_pred_gmm_deconf_unconf)
    pbar.set_postfix({"Component #": comp_k, "Accuracy": np.round(acc*100,2)})
    pbar.refresh()
    
    metrics_container.loc[comp_k, "Semi-synthetic classification task"] = acc
    metrics_container.to_csv(save_path + "metrics_container_simulations.csv", index=True, header=True)
pbar.close()
#%%

metrics_container = pd.read_csv("metrics_container.csv", index_col=0)
comp_k_lst = [2, 5, 10, 20, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]

fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()  

# -------------- y-axis (accuracy left) --------------
acc_syn   = 100 * metrics_container["Synthetic classification task"].values
acc_semi  = 100 * metrics_container["Semi-synthetic classification task"].values

line1, = ax1.plot(
    comp_k_lst, acc_syn,
    marker='X', markersize=10, linestyle='-',
    label='Acc. - Synthetic classification (SVM)', lw=2.5, color ='cornflowerblue'
)
line2, = ax1.plot(
    comp_k_lst, acc_semi,
    marker='X', markersize=10, linestyle='-',
    label='Acc. - Background-MNIST classification (KNN)', lw=2.5, color ='orange'
)
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy (%)', fontsize=16)
ax1.set_ylim(50, 100)

# -------------- y-axis (f1-score right) --------------
f1_real = metrics_container["ICH detection task"].values

line3, = ax2.plot(
    comp_k_lst, f1_real,
    marker='X', markersize=10, linestyle='-',
    label='F1-score - Real-world ICH detection (ResNet-CNN)', lw=2.5, color ='salmon'
)

ax2.set_xscale('log')
ax2.set_ylabel('F1-score', fontsize=16)
ax2.set_ylim(0.5, 1) \

# -------------- x-axis in log scale --------------
ax1.set_xscale('log')
ax1.xaxis.set_major_locator(FixedLocator(comp_k_lst))
ax1.xaxis.set_major_formatter(FuncFormatter(fmt_sci))
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

ax1.set_xlabel('Number of Components (K)', fontsize=16)

# Only draw grid on the main axis
ax1.grid(True, which='both', axis='both')

# Combine legends from both axes
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

fig.tight_layout()
fig.savefig('ML_model_metrics_over_comp_k_with_two_y_axes.png',
            dpi=600, bbox_inches='tight')
plt.show()
# %%
