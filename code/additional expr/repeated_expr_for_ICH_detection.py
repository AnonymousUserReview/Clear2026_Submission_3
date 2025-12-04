#%%
import os, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter('ignore')
from mechanism_learn import pipeline as mlpipe
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import os 
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from evaluator_utils import classification_eval
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
tf.get_logger().setLevel(logging.ERROR)
import gc
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

print("TensorFlow version:", tf.__version__)
print("Built with CUDA?:", tf.test.is_built_with_cuda())
print("Built with GPU?:", tf.test.is_built_with_gpu_support())
print("Available GPU device:", tf.config.list_physical_devices('GPU'))
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def img_read(dir_list, img_size):
    img_list = []
    for dir in dir_list:
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img_list.append(img.flatten())
    return np.array(img_list)

def resNetCNN_model(input_shape, num_class):
    input_img = layers.Input(shape=input_shape)
    
    short_cut = input_img
    x = layers.Conv2D(16, (7, 7), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2, 2), padding='same')(x)
    
    short_cut = layers.AveragePooling2D((8, 8), padding='same')(short_cut)
    x = layers.Add()([x, short_cut])
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    encoded = layers.Dense(num_class, activation='softmax')(x)
    
    return models.Model(input_img, encoded)
#%%

# Reading and preprocessing the ICH dataset
dir = r"../../test_data/ICH_data/"
effect_dir = dir + r"ct_clean/"
mediator_dir = dir 
cause_dir = dir
imgs_names = os.listdir(effect_dir)
imgs_names = sorted(imgs_names, key=lambda x: int(x.split('.')[0]))

effect_imgs = img_read([effect_dir + img_name for img_name in imgs_names], (128, 128))
cause_table = pd.read_csv(cause_dir + "hemorrhage_diagnosis_ct_clean.csv")
mediator_table = pd.read_csv(mediator_dir + "mediator_embedding.csv")

# Encoding the cause categories
cause_table["category"] = np.nan
cause_table.loc[cause_table["No_Hemorrhage"] == 1, "category"] = 0
cause_table.loc[cause_table["Intraparenchymal"] == 1, "category"] = 1
cause_table.loc[cause_table["Epidural"] == 1, "category"] = 2
cause_table.loc[cause_table["Subdural"] == 1, "category"] = 3
cause_table.loc[cause_table["Intraventricular"] == 1, "category"] = 4
cause_table.loc[cause_table["Subarachnoid"] == 1, "category"] = 5

# Dropping the redundant columns in the mediator table
mediator_table.drop(columns=['3'], inplace=True)

# Preparing the data for mechanism learning
cause_category = cause_table["category"].values
cause_category = cause_category.reshape(-1,1)
mediaor_values = mediator_table.values
n_class = len(cause_table["category"].unique())
X_d = effect_imgs.shape[1]
image_h = int(np.sqrt(X_d))
image_w = int(np.sqrt(X_d))

# Reducing the dimension of the effect images using PCA
img_pca = PCA(n_components=0.95)
img_pca.fit(effect_imgs)
effect_imgs_lowd_embedding = img_pca.transform(effect_imgs)
reduced_X_d = effect_imgs_lowd_embedding.shape[1]

# Splitting the data into training, validation, and test sets (confounded)
test_prop = 0.4
X_train_conf, X_testval_conf, Y_train_conf, Y_testval_conf = train_test_split(effect_imgs, cause_category, 
                                                                             test_size=test_prop, random_state=42, stratify=cause_category)

val_prop = 0.4
X_val_conf, X_test_conf, Y_val_conf, Y_test_conf = train_test_split(X_testval_conf, Y_testval_conf,
                                                                    test_size=1-val_prop, random_state=42, stratify=Y_testval_conf)

#%%
# Parameters for repeated experiments
expr_n = 30
# Parameters for resampling
n_train_sample = [5000 for i in range(len(np.unique(cause_category)))]
n_val_sample = np.unique(Y_val_conf, return_counts=True)[1]
n_test_sample = np.unique(Y_test_conf, return_counts=True)[1]
# Parameters for CWGMM
comp_k = [400, 10, 55, 11, 4, 3]
max_iter = 500
cov_reg = 1e-2
min_variance_value = 1e-2
tol = 1e-5
cov_type = "diag"
init_method = "kmeans++"
#%%
res_dir = r"../../res_table/"
save_filename = "ICH_detection_{}repeated_expr_result_".format(expr_n)
metrics_df = pd.DataFrame(columns=["acc mean", "prec mean", "recall mean", "f1 mean", "acc std", "prec std", "recall std", "f1 std"])
metrics_records_gmm_confTest = np.zeros((expr_n, 4))
metrics_records_gmm_unconfTest = np.zeros((expr_n, 4))
metrics_records_cb_confTest = np.zeros((expr_n, 4))
metrics_records_cb_unconfTest = np.zeros((expr_n, 4))
metrics_records_conf_confTest = np.zeros((expr_n, 4))
metrics_records_conf_unconfTest = np.zeros((expr_n, 4))

pbar = tqdm(range(expr_n), total=expr_n, desc = "Repeated experiments", unit="expr")
for expr_i in pbar:

    # Initializing the machanism learning pipeline
    ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = cause_category,
                                                        mechanism_data = mediaor_values, 
                                                        effect_data = effect_imgs_lowd_embedding, 
                                                        intv_values = np.unique(cause_category), 
                                                        dist_map = None, 
                                                        est_method = "kde",
                                                        bandwidth = "scott"
                                                        )

    # Fitting the CWGMM model
    ## Don't sample the data, just fit and return the CWGMM model for later sampling
    ## Set different comp_k for different intervention categories because of the class imbalance
    ml_gmm_pipeline.cwgmm_fit(comp_k = comp_k,
                            max_iter = max_iter, 
                            min_variance_value = min_variance_value,
                            cov_reg = cov_reg,
                            tol = tol, 
                            init_method = init_method, 
                            cov_type = cov_type, 
                            random_seed = None, 
                            return_model = False,
                            verbose = 0)

    # Sample the deconfounded training data
    X_train_deconf_gmm, Y_train_deconf_gmm = ml_gmm_pipeline.cwgmm_resample(n_samples=n_train_sample, return_samples = True)
    # Inverse transform the sampled image embedding to the original space
    X_train_deconf_gmm = img_pca.inverse_transform(X_train_deconf_gmm)
    # Clip the X values to be in the range of [0, 255] as the original images
    X_train_deconf_gmm = np.clip(X_train_deconf_gmm, 0, 255.0)
    # Reshape the data to the original image shape
    X_train_deconf_gmm = X_train_deconf_gmm.reshape(-1,image_h,image_w,1)

    X_val_deconf, Y_val_deconf = ml_gmm_pipeline.cwgmm_resample(n_samples=n_val_sample, return_samples = True)
    # Inverse transform the sampled image embedding to the original space
    X_val_deconf = img_pca.inverse_transform(X_val_deconf)
    # Clip the X values to be in the range of [0, 255] as the original images
    X_val_deconf = np.clip(X_val_deconf, 0, 255.0)
    # Reshape the data to the original image shape
    X_val_deconf = X_val_deconf.reshape(-1,image_h,image_w,1)

    X_test_deconf, Y_test_deconf = ml_gmm_pipeline.cwgmm_resample(n_samples=n_test_sample, return_samples = True)
    # Inverse transform the sampled image embedding to the original space
    X_test_deconf = img_pca.inverse_transform(X_test_deconf)
    # Clip the X values to be in the range of [0, 255] as the original images
    X_test_deconf = np.clip(X_test_deconf, 0, 255.0)
    # Reshape the data to the original image shape
    X_test_deconf = X_test_deconf.reshape(-1,image_h,image_w,1)

    ResNet_gmm_deconf = resNetCNN_model((image_h, image_w, 1), n_class)
    ResNet_gmm_deconf.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    Y_train_deconf_gmm_oh = to_categorical(Y_train_deconf_gmm.reshape(-1), num_classes=n_class)
    Y_val_deconf_oh = to_categorical(Y_val_deconf.reshape(-1), num_classes=n_class)

    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                                patience=15, verbose=0, mode='max',
                                baseline=None, restore_best_weights=True)

    ResNet_gmm_deconf.fit(X_train_deconf_gmm, Y_train_deconf_gmm_oh, 
                        epochs=75, batch_size=8, shuffle=True,
                        validation_data=(X_val_deconf, Y_val_deconf_oh),
                        callbacks=[early_stopping], verbose=0)

    Y_pred_deconfModel_confTest_gmm = ResNet_gmm_deconf.predict(X_test_conf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_deconfModel_deconfTest_gmm = ResNet_gmm_deconf.predict(X_test_deconf, verbose=0)
    Y_pred_deconfModel_confTest_gmm = np.argmax(Y_pred_deconfModel_confTest_gmm, axis=1)
    Y_pred_deconfModel_deconfTest_gmm = np.argmax(Y_pred_deconfModel_deconfTest_gmm, axis=1)

    tf.keras.backend.clear_session()
    gc.collect()

    # Initializing the machanism learning pipeline using CB-based deconfounding method
    ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = cause_category,
                                                    mechanism_data = mediaor_values, 
                                                    effect_data = effect_imgs_lowd_embedding, 
                                                    intv_values = np.unique(cause_category), 
                                                    dist_map = None, 
                                                    est_method = "kde",
                                                    bandwidth = "scott"
                                                    )
    # Resample the data using the front-door CB
    X_train_deconf_cb, Y_train_deconf_cb = ml_cb_pipeline.cb_resample(n_samples = n_train_sample,
                                                                      cb_mode = "fast",
                                                                      return_samples = True,
                                                                      verbose=0)

    # Inverse transform the sampled image embedding to the original space
    X_train_deconf_cb = img_pca.inverse_transform(X_train_deconf_cb)
    # Clip the X values to be in the range of [0, 255] as the original images
    X_train_deconf_cb = np.clip(X_train_deconf_cb, 0, 255.0)
    # Reshape the data to the original image shape
    X_train_deconf_cb = X_train_deconf_cb.reshape(-1,image_h,image_w,1)

    ResNet_cb_deconf = resNetCNN_model((image_h, image_w, 1), n_class)
    ResNet_cb_deconf.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    Y_train_deconf_cb_oh = to_categorical(Y_train_deconf_cb.reshape(-1), num_classes=n_class)
    Y_val_deconf_oh = to_categorical(Y_val_deconf.reshape(-1), num_classes=n_class)

    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                                patience=15, verbose=0, mode='max',
                                baseline=None, restore_best_weights=True)

    ResNet_cb_deconf.fit(X_train_deconf_cb, Y_train_deconf_cb_oh, 
                        epochs=75, batch_size=8, shuffle=True,
                        validation_data=(X_val_deconf, Y_val_deconf_oh),
                        callbacks=[early_stopping], verbose=0)

    Y_pred_deconfModel_confTest_cb = ResNet_cb_deconf.predict(X_test_conf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_deconfModel_deconfTest_cb = ResNet_cb_deconf.predict(X_test_deconf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_deconfModel_confTest_cb = np.argmax(Y_pred_deconfModel_confTest_cb, axis=1)
    Y_pred_deconfModel_deconfTest_cb = np.argmax(Y_pred_deconfModel_deconfTest_cb, axis=1)

    tf.keras.backend.clear_session()
    gc.collect()

    X_train_conf_resampled = np.empty((0, image_h*image_w))
    Y_train_conf_resampled = np.empty((0, 1))

    for class_i in range(n_class):
        idx_class_i = np.where(Y_train_conf == class_i)[0]
        random_sample_idx = np.random.choice(idx_class_i, n_train_sample[class_i], replace=True)
        X_train_conf_resampled = np.vstack((X_train_conf_resampled, X_train_conf[random_sample_idx]))
        Y_train_conf_resampled = np.vstack((Y_train_conf_resampled, Y_train_conf[random_sample_idx]))
    X_train_conf_oversample = X_train_conf_resampled.reshape(-1, image_h, image_w, 1)
    Y_train_conf_oversample = Y_train_conf_resampled.reshape(-1, 1)

    Y_train_conf_oh = to_categorical(Y_train_conf_oversample, num_classes=n_class)
    Y_val_conf_oh = to_categorical(Y_val_conf, num_classes=n_class)

    ResNet_conf = resNetCNN_model((image_h, image_w, 1), n_class)
    ResNet_conf.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                                patience=15, verbose=0, mode='max',
                                baseline=None, restore_best_weights=True)

    ResNet_conf.fit(X_train_conf_oversample, Y_train_conf_oh, 
                epochs=75, batch_size=8, shuffle=True,
                validation_data=(X_val_conf.reshape(-1, image_h, image_w, 1), Y_val_conf_oh),
                callbacks=[early_stopping], verbose=0)

    Y_pred_confModel_confTest = ResNet_conf.predict(X_test_conf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_confModel_deconfTest = ResNet_conf.predict(X_test_deconf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_confModel_confTest = np.argmax(Y_pred_confModel_confTest, axis=1)
    Y_pred_confModel_deconfTest = np.argmax(Y_pred_confModel_deconfTest, axis=1)

    tf.keras.backend.clear_session()
    gc.collect()
    
    evaluator_gmm_unconf = classification_eval(y_true = Y_test_deconf.reshape(-1), y_pred = Y_pred_deconfModel_deconfTest_gmm)
    metrics_records_gmm_unconfTest[expr_i] = evaluator_gmm_unconf.metrics(report=False, mode = 'weighted')
    evaluator_gmm_conf = classification_eval(y_true = Y_test_conf.reshape(-1), y_pred = Y_pred_deconfModel_confTest_gmm)
    metrics_records_gmm_confTest[expr_i] = evaluator_gmm_conf.metrics(report=False, mode = 'weighted')
    
    evaluator_cb_unconf = classification_eval(y_true = Y_test_deconf.reshape(-1), y_pred = Y_pred_deconfModel_deconfTest_cb)
    metrics_records_cb_unconfTest[expr_i] = evaluator_cb_unconf.metrics(report=False, mode = 'weighted')
    evaluator_cb_conf = classification_eval(y_true = Y_test_conf.reshape(-1), y_pred = Y_pred_deconfModel_confTest_cb)
    metrics_records_cb_confTest[expr_i] = evaluator_cb_conf.metrics(report=False, mode = 'weighted')
    
    evaluator_conf_unconf = classification_eval(y_true = Y_test_deconf.reshape(-1), y_pred = Y_pred_confModel_deconfTest)
    metrics_records_conf_unconfTest[expr_i] = evaluator_conf_unconf.metrics(report=False, mode = 'weighted')    
    evaluator_conf_conf = classification_eval(y_true = Y_test_conf.reshape(-1), y_pred = Y_pred_confModel_confTest)
    metrics_records_conf_confTest[expr_i] = evaluator_conf_conf.metrics(report=False, mode = 'weighted')
 
    pbar.set_postfix({"gmm unconf test F1": metrics_records_gmm_unconfTest[expr_i][3],
                      "cb unconf test F1": metrics_records_cb_unconfTest[expr_i][3],
                      "conf unconf test F1": metrics_records_conf_unconfTest[expr_i][3]})
#%% Computing the mean and std of the metrics
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

metrics_df.loc["CW-GMM-based deconfounded model (unconfounded test)"] = np.concatenate([metrics_mean_gmm_unconfTest, metrics_std_gmm_unconfTest])
metrics_df.loc["CB-based deconfounded model (unconfounded test)"] = np.concatenate([metrics_mean_cb_unconfTest, metrics_std_cb_unconfTest])
metrics_df.loc["Confounded model (unconfounded test)"] = np.concatenate([metrics_mean_conf_unconfTest, metrics_std_conf_unconfTest])

metrics_df.loc["CW-GMM-based deconfounded model (confounded test)"] = np.concatenate([metrics_mean_gmm_confTest, metrics_std_gmm_confTest])
metrics_df.loc["CB-based deconfounded model (confounded test)"] = np.concatenate([metrics_mean_cb_confTest, metrics_std_cb_confTest])
metrics_df.loc["Confounded model (confounded test)"] = np.concatenate([metrics_mean_conf_confTest, metrics_std_conf_confTest])

# Save results
cur_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
metrics_df.to_csv(res_dir + save_filename + cur_datetime + ".csv")
# %%
