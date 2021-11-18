import logging
from nn_globals import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class evaluate:
    """
    To evalute and assess model performance against the test data.
    Inputs:
    X_test -> Input numpy array of features.
    y_test -> List of numpy array representing predicted variables, momentum,
    and displacement.
    """
    def __init__(self,X_test,y_test):
        self.X = X_test
        self.y = y_test[0]
        self.dxy = y_test[1]

    def compute_data_statistics(self,ctype = "y",label="data"):
        if ctype == "y":
            x = self.recalibrate(self.y,reg_pt_scale)
            x = x**(-1)
        else:
            x = self.recalibrate(self.dxy,reg_dxy_scale)
        df_describe = pd.DataFrame(x, columns = [label])
        print(df_describe.describe())

    def rmse(self,y_true, y_predicted):
        assert(y_true.shape[0] == y_predicted.shape[0])
        n = y_true.shape[0]
        sum_square = np.sum((y_true - y_predicted)**2)
        return math.sqrt(sum_square/n)

    def adjusted_r_2(self,y_true, y_predicted):
        y_addC = sm.add_constant(y_true)
        result = sm.OLS(y_predicted, y_addC).fit()
        print(result.rsquared, result.rsquared_adj)

    def recalibrate(self,x,scale):
        return x/scale

    def inverse(self,arr):
        arr_inv = 1./arr
        arr_inv[arr_inv == np.inf] = 0.
        return arr_inv

    def predict(self,model,batch_size = 2000):
        y_test_true = self.recalibrate(self.y,reg_pt_scale)
        dxy_test_true = self.recalibrate(self.dxy, reg_dxy_scale)

        y_test = model.predict(self.X,batch_size = batch_size)
        y_test_meas = y_test[:,0]
        dxy_test_meas = y_test[:,1]
        y_test_meas = self.recalibrate(y_test_meas,reg_pt_scale)
        dxy_test_meas = self.recalibrate(dxy_test_meas,reg_dxy_scale)

        y_test_meas = y_test_meas.reshape(-1)
        dxy_test_meas = dxy_test_meas.reshape(-1)

        return y_test_meas, dxy_test_meas

#     def compute_error(self,y_predicted,ctype = "y"):
#         if ctype == "y":
#             y_test_true = self.recalibrate(self.y,reg_pt_scale)
#             print("RMSE Error for momentum:",self.rmse(self.inverse(y_test_true),\
#                                                                               self.inverse(y_predicted)))
#         else:
#             dxy_test_true = self.recalibrate(self.dxy, reg_dxy_scale)
#             print("RMSE Error for dxy:",self.rmse(dxy_test_true,y_predicted))

    def get_error(self,y_predicted,ctype = "y"):
        if ctype == "y":
            y_test_true = self.recalibrate(self.y,reg_pt_scale)
            return self.rmse(self.inverse(y_test_true),self.inverse(y_predicted))
        else:
            dxy_test_true = self.recalibrate(self.dxy, reg_dxy_scale)
            return self.rmse(dxy_test_true,y_predicted)

def k_fold_validation(model, x, y, dxy, folds =10, eval_batch_size = 2000):
    """
    Function to evaluate and print the error over the data by dividing it into k folds using random selection
    and computing an average of the results.
    Inputs:
    model -> The trained model
    x -> Set of input features/ predictors
    y -> First Predicted Value (angular momentum)
    dxy -> Second Predicted Value (displacement)
    folds -> Default = 10, the number of batches into which the x,y,dxy are to be divided
    eval_batch_size -> Default set to 2000, the batch size for the data to evaluate the results (generally, kept
    > than the training batch size)
    Returns: None

    """
    x_copy = np.copy(x)
    y_copy = np.copy(y)
    dxy_copy = np.copy(dxy)

    assert x_copy.shape[0] == y_copy.shape[0] == dxy_copy.shape[0]

    fold_size = int(x_copy.shape[0] / folds)
    x_splits, y_splits, dxy_splits = [], [], []

    for i in range(folds):
        indices = np.random.choice(x_copy.shape[0],fold_size, replace=False)
        x_splits.append(x_copy[indices])
        y_splits.append(y_copy[indices])
        dxy_splits.append(dxy_copy[indices])
        x_copy = np.delete(x_copy,indices,axis = 0)
        y_copy = np.delete(y_copy,indices,axis = 0)
        dxy_copy = np.delete(dxy_copy,indices,axis = 0)
    rmse_y, rmse_dxy = [],[]
    for i in range(folds):
        evaluate_obj = evaluate(x_splits[i], tuple([y_splits[i],dxy_splits[i]]))
        y_predicted , dxy_predicted = evaluate_obj.predict(model = model,batch_size = eval_batch_size)
        rmse_y.append(evaluate_obj.get_error(y_predicted,ctype="y"))
        rmse_dxy.append(evaluate_obj.get_error(dxy_predicted,ctype="dxy"))

    print('Average RMSE for '+ str(folds) + '-fold cv for y:', np.mean(rmse_y))
    print('Average RMSE for '+ str(folds) + '-fold cv for dxy:', np.mean(rmse_dxy))


def huber_loss(y_true, y_pred, delta=1.345):
    """
    Function to compute the huber loss.
    Inputs:
    y_true -> The truth values for the predicted variable
    y_pred -> The values predcited by the model for the output variable
    delta -> Threshold for computing MSE or MAE, > delta, MSE is the chosen loss,
    else, MAE is the loss value returned.
    Returns:
    Loss Tensor.
    """
    x = K.abs(y_true - y_pred)
    squared_loss = 0.5*K.square(x)
    absolute_loss = delta * (x - 0.5*delta)
    #xx = K.switch(x < delta, squared_loss, absolute_loss)
    xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow
    return K.mean(xx, axis=-1)


def get_sparsity(weights):
    """
    Code borrowed from https://github.com/google/qkeras/blob/master/qkeras/utils.py#L937
    Inputs:
    Numpy array of weights of the keras model
    Returns:
    Sparsity as the ratio of non-zero weights to the total weights within the weights matrix.
    """
    try:
        if isinstance(a, np.ndarray):
            return 1.0 - np.count_nonzero(weights) / float(weights.size)
    except:
        print("The input weights must be an array!")
