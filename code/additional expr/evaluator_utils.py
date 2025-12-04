# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class classification_eval:
    
    def __init__(self, y_true, X =None, model = None, y_pred=None):
        self.X = X
        self.y_true = y_true
        if model is not None:
            self.model = model
            self.y_pred = self.generate_pred()
        elif y_pred is not None:
            self.y_pred = y_pred
        else:
            raise ValueError("Either model or y_pred must be provided.")
    
    def generate_pred(self):
        return self.model.predict(self.X)
    
    # Metrics calculation and report printing    
    def metrics(self, report=False, mode = 'macro', report_prefix=''):
        multiclass = len(list(set(self.y_true))) > 2
        # for multiclass classification, use macro-average
        if multiclass:
            acc = accuracy_score(self.y_true, self.y_pred)
            prec = precision_score(self.y_true, self.y_pred,average = mode)
            recall = recall_score(self.y_true, self.y_pred,average = mode)
            f1score = f1_score(self.y_true, self.y_pred,average = mode)
        # for binary classification, directly compute
        else:
            acc = accuracy_score(self.y_true, self.y_pred)
            prec = precision_score(self.y_true, self.y_pred)
            recall = recall_score(self.y_true, self.y_pred)
            f1score = f1_score(self.y_true, self.y_pred)
        # print out report
        if report:
            print("-"*50)
            print(report_prefix+" Acc: %.4f" %(acc))
            print(report_prefix+" Precision: %.4f" %(prec))
            print(report_prefix+" recall: %.4f" %(recall))
            print(report_prefix+" f1-score: %.4f" %(f1score))
            print("-"*50)
            
        return acc,prec,recall,f1score
    
    # Confusion matrix drawing   
    def conf_matrix(self, fig_name="Confision Matrix", axis_rename=None):
        confMat=confusion_matrix(self.y_true, self.y_pred)
        confMat=pd.DataFrame(confMat)
        if axis_rename==None:
            None
        else:
            confMat=confMat.rename(index=axis_rename,columns=axis_rename)
        plt.figure(num=fig_name, facecolor='lightgray')
        plt.title(fig_name, fontsize=20)
        ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
        ax.set_xlabel('Predicted Class', fontsize=14)
        ax.set_ylabel('True Class', fontsize=14)
        plt.show()        
        
class regression_eval:
    
    def __init__(self, y_true, X = None, model = None, y_pred=None):
        self.X = X
        self.y_true = y_true
        if model is not None:
            self.model = model
            self.y_pred = self.generate_pred()
        elif y_pred is not None:
            self.y_pred = y_pred
        else:
            raise ValueError("Either model or y_pred must be provided.")
        
    def generate_pred(self):
        return self.model.predict(self.X)
        
    def metrics(self, report=False, report_prefix=''):
        
        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        if report:
            print("-"*50)
            print(report_prefix+"MSE: %.6f" %(mse))
            print(report_prefix+"MAE: %.6f" %(mae))
            print(report_prefix+"MAPE: %.6f" %(mape))
            print(report_prefix+"R-square: %.6f" %(r2))
            print("-"*50) 
        
        return mse, mae, mape, r2
    
    def true_pred_plot(self, x_tickers):
        
        x_tickers = np.array(x_tickers)
        
        plt.figure()
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16  
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"
        plt.plot(x_tickers, self.y_pred,  label='Prediction')
        plt.plot(x_tickers, self.y_true,  label='Ground True')
        plt.ylim(min([min(self.y_pred), min(self.y_true)])*0.95, max([max(self.y_pred), max(self.y_true)])*1.05)
        plt.xlim(x_tickers[0],x_tickers[-1])
        plt.legend()
        plt.title("Predictions v.s Ground True")
        plt.show()