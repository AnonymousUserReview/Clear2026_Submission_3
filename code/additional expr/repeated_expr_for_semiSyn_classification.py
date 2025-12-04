#%%
from mechanism_learn import pipeline as mlpipe
from evaluator_utils import classification_eval
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import maximum_filter
import warnings
warnings.filterwarnings("ignore")
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

#%%
# Read semi-synthetic data
semisyn_data_dir = r"../../test_data/semi_synthetic_data/"
X_train_conf = pd.read_csv(semisyn_data_dir + "X_train_conf.csv").to_numpy()
Y_train_conf = pd.read_csv(semisyn_data_dir + "Y_train_conf.csv").to_numpy().ravel()
Z_train_conf = pd.read_csv(semisyn_data_dir + "Z_train_conf.csv").to_numpy()
X_train_conf = maxPooling_imgArr(X_train_conf, kernel_size=3, flatten=True)

X_train_unconf = pd.read_csv(semisyn_data_dir + "X_train_unconf.csv").to_numpy()
Y_train_unconf = pd.read_csv(semisyn_data_dir + "Y_train_unconf.csv").to_numpy().ravel()
X_train_unconf = maxPooling_imgArr(X_train_unconf, kernel_size=3, flatten=True)

X_test_unconf = pd.read_csv(semisyn_data_dir + "X_test_unconf.csv").to_numpy()
Y_test_unconf = pd.read_csv(semisyn_data_dir + "Y_test_unconf.csv").to_numpy().ravel()
X_test_unconf = maxPooling_imgArr(X_test_unconf, kernel_size=3, flatten=True)

X_test_conf = pd.read_csv(semisyn_data_dir + "X_test_conf.csv").to_numpy()
Y_test_conf = pd.read_csv(semisyn_data_dir + "Y_test_conf.csv").to_numpy().ravel()
X_test_conf = maxPooling_imgArr(X_test_conf, kernel_size=3, flatten=True)
#%%
# Parameters for repeated experiments
expr_n = 2
# Parameters for resampling
n_samples = [(Y_train_conf == 1).sum()*10, (Y_train_conf == 2).sum()*10]
# Parameters for CWGMM
comp_k = 300
max_iter = 500
cov_reg = 1e-3
min_variance_value = 2e-3
tol = 1e-2
cov_type = "diag"
# Parameters for weights estimation
est_method = "histogram"
n_bins = [0, 0]
#%%
res_dir = r"../../res_table/"
save_filename = "semiSyn_classification_{}repeated_expr_results_".format(expr_n)
metrics_df = pd.DataFrame(columns=["acc mean", "prec mean", "recall mean", "f1 mean", "acc std", "prec std", "recall std", "f1 std"])
metrics_records_gmm_confTest = np.zeros((expr_n, 4))
metrics_records_gmm_unconfTest = np.zeros((expr_n, 4))
metrics_records_cb_confTest = np.zeros((expr_n, 4))
metrics_records_cb_unconfTest = np.zeros((expr_n, 4))
metrics_records_conf_confTest = np.zeros((expr_n, 4))
metrics_records_conf_unconfTest = np.zeros((expr_n, 4))
metrics_records_unconf_confTest = np.zeros((expr_n, 4))
metrics_records_unconf_unconfTest = np.zeros((expr_n, 4))

pbar = tqdm(range(expr_n), total=expr_n, desc = "Repeated experiments", unit="expr")
for expr_i in pbar:
    # Mechanism learning-based deconfounded model
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
                              cov_type = cov_type,
                              cov_reg = cov_reg,
                              min_variance_value = min_variance_value,
                              return_model = False,
                              verbose = 0)
    ml_gmm_pipeline.cwgmm_resample(n_samples = n_samples,
                                   return_samples = True)
    deconf_gmm_clf = ml_gmm_pipeline.deconf_model_fit(ml_model = KNeighborsClassifier(n_neighbors = 3))

    # CB-based deconfounded model
    ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                       mechanism_data = Z_train_conf, 
                                                       effect_data = X_train_conf, 
                                                       intv_values = np.unique(Y_train_conf), 
                                                       est_method = est_method, 
                                                       n_bins = n_bins)
    ml_cb_pipeline.cb_resample(n_samples = n_samples,
                               return_samples = False,
                               verbose = 0)
    deconf_cb_clf = ml_cb_pipeline.deconf_model_fit(ml_model = KNeighborsClassifier(n_neighbors = 3))
    
    # Confounded model
    conf_clf = KNeighborsClassifier(n_neighbors = 3)
    conf_clf.fit(X_train_conf, Y_train_conf)
    
    # Unconfounded model
    unconf_clf = KNeighborsClassifier(n_neighbors = 3)
    unconf_clf.fit(X_train_unconf, Y_train_unconf)
    
    # Evaluate models
    evaluator_gmm_unconf = classification_eval(X = X_test_unconf, y_true = Y_test_unconf.reshape(-1), model = deconf_gmm_clf)
    metrics_records_gmm_unconfTest[expr_i] = evaluator_gmm_unconf.metrics(report=False)
    evaluator_gmm_conf = classification_eval(X = X_test_conf, y_true = Y_test_conf.reshape(-1), model = deconf_gmm_clf)
    metrics_records_gmm_confTest[expr_i] = evaluator_gmm_conf.metrics(report=False)
    
    evaluator_cb_unconf = classification_eval(X = X_test_unconf, y_true = Y_test_unconf.reshape(-1), model = deconf_cb_clf)
    metrics_records_cb_unconfTest[expr_i] = evaluator_cb_unconf.metrics(report=False)
    evaluator_cb_conf = classification_eval(X = X_test_conf, y_true = Y_test_conf.reshape(-1), model = deconf_cb_clf)
    metrics_records_cb_confTest[expr_i] = evaluator_cb_conf.metrics(report=False)
    
    evaluator_conf_conf = classification_eval(X = X_test_conf, y_true = Y_test_conf.reshape(-1), model = conf_clf)
    metrics_records_conf_confTest[expr_i] = evaluator_conf_conf.metrics(report=False)
    evaluator_conf_unconf = classification_eval(X = X_test_unconf, y_true = Y_test_unconf.reshape(-1), model = conf_clf)
    metrics_records_conf_unconfTest[expr_i] = evaluator_conf_unconf.metrics(report=False)    
    
    evaluator_unconf_conf = classification_eval(X = X_test_conf, y_true = Y_test_conf.reshape(-1), model = unconf_clf)
    metrics_records_unconf_confTest[expr_i] = evaluator_unconf_conf.metrics(report=False)
    evaluator_unconf_unconf = classification_eval(X = X_test_unconf, y_true = Y_test_unconf.reshape(-1), model = unconf_clf)
    metrics_records_unconf_unconfTest[expr_i] = evaluator_unconf_unconf.metrics(report=False)   
    
    pbar.set_postfix({"gmm unconf test acc": metrics_records_gmm_unconfTest[expr_i][0],
                      "cb unconf test acc": metrics_records_cb_unconfTest[expr_i][0],
                      "unconf unconf test acc": metrics_records_unconf_unconfTest[expr_i][0]})

#%% Compute mean and std of metrics
metrics_mean_gmm_confTest = np.round(np.mean(metrics_records_gmm_confTest, axis=0), 4)
metrics_std_gmm_confTest = np.round(np.std(metrics_records_gmm_confTest, axis=0), 4)
metrics_mean_gmm_unconfTest = np.round(np.mean(metrics_records_gmm_unconfTest, axis=0), 4)
metrics_std_gmm_unconfTest = np.round(np.std(metrics_records_gmm_unconfTest, axis=0), 4)

metrics_mean_cb_confTest = np.round(np.mean(metrics_records_cb_confTest, axis=0), 4)
metrics_std_cb_confTest = np.round(np.std(metrics_records_cb_confTest, axis=0), 4)
metrics_mean_cb_unconfTest = np.round(np.mean(metrics_records_cb_unconfTest, axis=0), 4)
metrics_std_cb_unconfTest = np.round(np.std(metrics_records_cb_unconfTest, axis=0), 4)

metrics_mean_conf_confTest = np.round(np.mean(metrics_records_conf_confTest, axis=0), 4)
metrics_std_conf_confTest = np.round(np.std(metrics_records_conf_confTest, axis=0), 4)
metrics_mean_conf_unconfTest = np.round(np.mean(metrics_records_conf_unconfTest, axis=0), 4)
metrics_std_conf_unconfTest = np.round(np.std(metrics_records_conf_unconfTest, axis=0), 4)

metrics_mean_unconf_confTest = np.round(np.mean(metrics_records_unconf_confTest, axis=0), 4)
metrics_std_unconf_confTest = np.round(np.std(metrics_records_unconf_confTest, axis=0), 4)
metrics_mean_unconf_unconfTest = np.round(np.mean(metrics_records_unconf_unconfTest, axis=0), 4)
metrics_std_unconf_unconfTest = np.round(np.std(metrics_records_unconf_unconfTest, axis=0), 4)    

metrics_df.loc["CW-GMM-based deconfounded model (unconfounded test)"] = np.concatenate([metrics_mean_gmm_unconfTest, metrics_std_gmm_unconfTest])
metrics_df.loc["CB-based deconfounded model (unconfounded test)"] = np.concatenate([metrics_mean_cb_unconfTest, metrics_std_cb_unconfTest])
metrics_df.loc["Confounded model (unconfounded test)"] = np.concatenate([metrics_mean_conf_unconfTest, metrics_std_conf_unconfTest])
metrics_df.loc["Unconfounded model (unconfounded test)"] = np.concatenate([metrics_mean_unconf_unconfTest, metrics_std_unconf_unconfTest])

metrics_df.loc["CW-GMM-based deconfounded model (confounded test)"] = np.concatenate([metrics_mean_gmm_confTest, metrics_std_gmm_confTest])
metrics_df.loc["CB-based deconfounded model (confounded test)"] = np.concatenate([metrics_mean_cb_confTest, metrics_std_cb_confTest])
metrics_df.loc["Confounded model (confounded test)"] = np.concatenate([metrics_mean_conf_confTest, metrics_std_conf_confTest])
metrics_df.loc["Unconfounded model (confounded test)"] = np.concatenate([metrics_mean_unconf_confTest, metrics_std_unconf_confTest])
# Save results
cur_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
metrics_df.to_csv(res_dir + save_filename + cur_datetime + ".csv")
# %%
