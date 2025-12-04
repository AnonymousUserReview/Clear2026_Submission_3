#%%
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from mechanism_learn import pipeline as mlpipe
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import cv2
import os 
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
tf.get_logger().setLevel(logging.ERROR)
import gc

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
# CWGMM parameters
max_iter = 500
cov_reg = 1e-2
min_variance_value = 1e-2
tol = 1e-5
cov_type = "diag"
comp_k_lst = [2, 5, 10, 20, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
initial_mean_method = "kmeans++"
# Data paths
dir = r"../../test_data/ICH_data/"
effect_dir = dir + r"ct_clean/"
mediator_dir = dir 
cause_dir = dir
imgs_names = os.listdir(effect_dir)
imgs_names = sorted(imgs_names, key=lambda x: int(x.split('.')[0]))

# Data split parameters
test_prop = 0.4
val_prop = 0.4
# Initialize or load the results dataframe
save_path = r"./results/"
save_filename = "metrics_container_ich_new.csv"
metrics_avg_strategy  = "weighted"  # "weighted" or "macro"
try:
    metrics_container = pd.read_csv(save_path + save_filename, index_col=0)
except:
    metrics_container = pd.DataFrame(index=comp_k_lst, columns=["Confounded test accuracy", 
                                                             "Non-confounded test accuracy", 
                                                             "Confounded test f1-score (weighted)", 
                                                             "Non-confounded test f1-score (weighted)"])
    metrics_container.to_csv(save_path + save_filename, index=True)

# %%
effect_imgs = img_read([effect_dir + img_name for img_name in imgs_names], (128, 128))
cause_table = pd.read_csv(cause_dir + "hemorrhage_diagnosis_ct_clean.csv")
mediator_table = pd.read_csv(mediator_dir + "mediator_embedding.csv")

cause_table["category"] = np.nan
cause_table.loc[cause_table["No_Hemorrhage"] == 1, "category"] = 0
cause_table.loc[cause_table["Intraparenchymal"] == 1, "category"] = 1
cause_table.loc[cause_table["Epidural"] == 1, "category"] = 2
cause_table.loc[cause_table["Subdural"] == 1, "category"] = 3
cause_table.loc[cause_table["Intraventricular"] == 1, "category"] = 4
cause_table.loc[cause_table["Subarachnoid"] == 1, "category"] = 5

mediator_table.drop(columns=['3'], inplace=True)

cause_category = cause_table["category"].values
cause_category = cause_category.reshape(-1,1)
mediator_values = mediator_table.values
n_class = len(cause_table["category"].unique())
X_d = effect_imgs.shape[1]
image_h = int(np.sqrt(X_d))
image_w = int(np.sqrt(X_d))

img_pca = PCA(n_components=0.95)
img_pca.fit(effect_imgs)
effect_imgs_lowd_embedding = img_pca.transform(effect_imgs)
reduced_X_d = effect_imgs_lowd_embedding.shape[1]

X_train_conf, X_testval_conf, Y_train_conf, Y_testval_conf = train_test_split(effect_imgs, cause_category, 
                                                                             test_size=test_prop, random_state=42, stratify=cause_category)

X_val_conf, X_test_conf, Y_val_conf, Y_test_conf = train_test_split(X_testval_conf, Y_testval_conf,
                                                                    test_size=1-val_prop, random_state=42, stratify=Y_testval_conf)

print("Sample the synthetic non-confounded validation and test data...")
# Initializing the machanism learning pipeline
ml_gmm_pipeline_nonconf_syn = mlpipe.mechanism_learning_process(cause_data = cause_category,
                                                                mechanism_data = mediator_values, 
                                                                effect_data = effect_imgs_lowd_embedding, 
                                                                intv_values = np.unique(cause_category), 
                                                                dist_map = None, 
                                                                est_method = "kde",
                                                                bandwidth = "scott"
                                                                )

# Fitting the CWGMM model
## Don't sample the data, just fit and return the CWGMM model for later sampling
## Set different comp_k for different intervention categories because of the class imbalance
ml_gmm_pipeline_nonconf_syn.cwgmm_fit(  comp_k = [400, 10, 55, 11, 4, 3],
                                        max_iter = max_iter, 
                          				tol = tol,
                                        cov_reg = cov_reg,
                                        min_variance_value = min_variance_value,
                         				init_method = initial_mean_method, 
                          				cov_type = cov_type, 
                          				random_seed = None,
		          						verbose = 0)

n_train_sample = [5000 for i in range(len(np.unique(cause_category)))]
n_val_sample = np.unique(Y_val_conf, return_counts=True)[1]
n_test_sample = np.unique(Y_test_conf, return_counts=True)[1]

X_val_deconf, Y_val_deconf = ml_gmm_pipeline_nonconf_syn.cwgmm_resample(n_samples=n_val_sample, return_samples = True)
# Inverse transform the sampled image embedding to the original space
X_val_deconf = img_pca.inverse_transform(X_val_deconf)
# Clip the X values to be in the range of [0, 255] as the original images
X_val_deconf = np.clip(X_val_deconf, 0, 255.0)
# Reshape the data to the original image shape
X_val_deconf = X_val_deconf.reshape(-1,image_h,image_w,1)

X_test_deconf, Y_test_deconf = ml_gmm_pipeline_nonconf_syn.cwgmm_resample(n_samples=n_test_sample, return_samples = True)
# Inverse transform the sampled image embedding to the original space
X_test_deconf = img_pca.inverse_transform(X_test_deconf)
# Clip the X values to be in the range of [0, 255] as the original images
X_test_deconf = np.clip(X_test_deconf, 0, 255.0)
# Reshape the data to the original image shape
X_test_deconf = X_test_deconf.reshape(-1,image_h,image_w,1)
print("Synthetic non-confounded datasets are generated!")

#%%
pbar = tqdm(enumerate(comp_k_lst), total = len(comp_k_lst), desc="Exprs. running...", dynamic_ncols=True)
for i, comp_k in pbar:
    # Initializing the mechanism learning pipeline
    ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = cause_category,
                                                        mechanism_data = mediator_values, 
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
                                   tol = tol, 
                                   init_method = initial_mean_method, 
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
    
    ResNet_gmm_deconf = resNetCNN_model((image_h, image_w, 1), n_class)
    ResNet_gmm_deconf.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    Y_train_deconf_gmm_oh = to_categorical(Y_train_deconf_gmm.reshape(-1), num_classes=n_class)
    Y_val_deconf_oh = to_categorical(Y_val_deconf.reshape(-1), num_classes=n_class)

    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0.001,
                                patience=8, verbose=0, mode='max',
                                baseline=None, restore_best_weights=True)

    ResNet_gmm_deconf.fit(X_train_deconf_gmm, Y_train_deconf_gmm_oh, 
                        epochs=60, batch_size=4, shuffle=True,
                        validation_data=(X_val_deconf, Y_val_deconf_oh),
                        callbacks=[early_stopping], verbose = 0)
    
    Y_pred_deconfModel_deconfTest_gmm = ResNet_gmm_deconf.predict(X_test_deconf, verbose=0)
    Y_pred_deconfModel_deconfTest_gmm = np.argmax(Y_pred_deconfModel_deconfTest_gmm, axis=1)
    
    Y_pred_deconfModel_confTest_gmm = ResNet_gmm_deconf.predict(X_test_conf.reshape(-1, image_h, image_w, 1), verbose=0)
    Y_pred_deconfModel_confTest_gmm = np.argmax(Y_pred_deconfModel_confTest_gmm, axis=1)
    
    acc_nonconf = accuracy_score(Y_test_deconf, Y_pred_deconfModel_deconfTest_gmm)
    acc_conf = accuracy_score(Y_test_conf, Y_pred_deconfModel_confTest_gmm)
    f1_nonconf = f1_score(Y_test_deconf, Y_pred_deconfModel_deconfTest_gmm, average=metrics_avg_strategy)
    f1_conf = f1_score(Y_test_conf, Y_pred_deconfModel_confTest_gmm, average=metrics_avg_strategy)
    
    pbar.set_postfix({"Component #": comp_k, "Non-conf f1": np.round(f1_nonconf,3), "Conf f1": np.round(f1_conf,3)})
    pbar.refresh()

    metrics_container.loc[comp_k, "Non-confounded test accuracy"] = acc_nonconf
    metrics_container.loc[comp_k, "Confounded test accuracy"] = acc_conf
    metrics_container.loc[comp_k, "Non-confounded test f1-score (weighted)"] = f1_nonconf
    metrics_container.loc[comp_k, "Confounded test f1-score (weighted)"] = f1_conf
    
    metrics_container.to_csv(save_path + save_filename, index=True, header=True)
    
    tf.keras.backend.clear_session()
    gc.collect()
pbar.close()
# %%
