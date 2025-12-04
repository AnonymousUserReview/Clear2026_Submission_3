#%%
from mechanism_learn import pipeline as mlpipe
from evaluator_utils import regression_eval
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
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

#%%

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

# random sample 10000 from training set for faster training
idx_conf = np.random.choice(len(X_train_conf), 10000, replace=False)
X_train_conf = X_train_conf[idx_conf]
Y_train_conf = Y_train_conf[idx_conf]
Z_train_conf = Z_train_conf[idx_conf]

idx_unconf = np.random.choice(len(X_train_unconf), 10000, replace=False)
X_train_unconf = X_train_unconf[idx_unconf]
Y_train_unconf = Y_train_unconf[idx_unconf]

# random sample 5000 from testing set for faster evaluation
idx_conf_test = np.random.choice(len(X_test_conf), 5000, replace=False)
X_test_conf = X_test_conf[idx_conf_test]
Y_test_conf = Y_test_conf[idx_conf_test]

idx_unconf_test = np.random.choice(len(X_test_unconf), 5000, replace=False)
X_test_unconf = X_test_unconf[idx_unconf_test]
Y_test_unconf = Y_test_unconf[idx_unconf_test]

#%%
# Parameters for repeated experiments
expr_n = 100
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
# Parameters for resampling
intv_intval_num = 50
Y_interv_values = np.linspace(Y_train_conf.min()*1.6, Y_train_conf.max()*1.3, intv_intval_num)
n_samples = [int(N // intv_intval_num)] * intv_intval_num
# Declare results storage
res_dir = r"../../res_table/"
save_filename = "syn_regression_{}repeated_expr_result_".format(expr_n)
metrics_df = pd.DataFrame(columns=["mse mean", 
                                   "mae mean", 
                                   "mape mean", 
                                   "R2 mean", 
                                   "mse std", 
                                   "mae std", 
                                   "mape std", 
                                   "R2 std"])
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
                              min_variance_value = min_variance_value,
                              verbose = 0,
                              return_model = False)
    ml_gmm_pipeline.cwgmm_resample(n_samples = n_samples,
                                   return_samples = False)
    deconf_gmm_lr = ml_gmm_pipeline.deconf_model_fit(ml_model = LinearRegression())
    
    # CB-based deconfounded model
    ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                       mechanism_data = Z_train_conf, 
                                                       effect_data = X_train_conf, 
                                                       intv_values = Y_interv_values, 
                                                       dist_map = None, 
                                                       est_method = est_method
                                                       )
    ml_cb_pipeline.cb_resample(n_samples = n_samples,
                               return_samples = False,
                               verbose = 0)
    deconf_cb_lr = ml_cb_pipeline.deconf_model_fit(ml_model = LinearRegression())
    
    # Confounded model
    conf_lr = LinearRegression()
    conf_lr = conf_lr.fit(X_train_conf, Y_train_conf.reshape(-1))
    
    # Unconfounded model
    unconf_lr = LinearRegression()
    unconf_lr = unconf_lr.fit(X_train_unconf, Y_train_unconf.reshape(-1))
    
    # Evaluate models
    evaluator_gmm_unconf = regression_eval(X = X_test_unconf, y_true = Y_test_unconf, model = deconf_gmm_lr)
    metrics_records_gmm_unconfTest[expr_i] = evaluator_gmm_unconf.metrics(report=False)
    evaluator_gmm_conf = regression_eval(X = X_test_conf, y_true = Y_test_conf, model = deconf_gmm_lr)
    metrics_records_gmm_confTest[expr_i] = evaluator_gmm_conf.metrics(report=False)
    
    evaluator_cb_unconf = regression_eval(X = X_test_unconf, y_true = Y_test_unconf, model = deconf_cb_lr)
    metrics_records_cb_unconfTest[expr_i] = evaluator_cb_unconf.metrics(report=False)
    evaluator_cb_conf = regression_eval(X = X_test_conf, y_true = Y_test_conf, model = deconf_cb_lr)
    metrics_records_cb_confTest[expr_i] = evaluator_cb_conf.metrics(report=False)
    
    evaluator_conf_conf = regression_eval(X = X_test_conf, y_true = Y_test_conf, model = conf_lr)
    metrics_records_conf_confTest[expr_i] = evaluator_conf_conf.metrics(report=False)
    evaluator_conf_unconf = regression_eval(X = X_test_unconf, y_true = Y_test_unconf, model = conf_lr)
    metrics_records_conf_unconfTest[expr_i] = evaluator_conf_unconf.metrics(report=False)
    
    evaluator_unconf_conf = regression_eval(X = X_test_conf, y_true = Y_test_conf, model = unconf_lr)
    metrics_records_unconf_confTest[expr_i] = evaluator_unconf_conf.metrics(report=False)
    evaluator_unconf_unconf = regression_eval(X = X_test_unconf, y_true = Y_test_unconf, model = unconf_lr)
    metrics_records_unconf_unconfTest[expr_i] = evaluator_unconf_unconf.metrics(report=False)
    
    pbar.set_postfix({"gmm unconf test mse": metrics_records_gmm_unconfTest[expr_i][0],
                      "cb unconf test mse": metrics_records_cb_unconfTest[expr_i][0]})

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
