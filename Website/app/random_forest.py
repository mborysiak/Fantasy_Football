from sklearn.linear_model import LinearRegression
import numpy as np

def fit_model(val):

    X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
    y = np.array([5, 10, 11, 13, 15])
    lr = LinearRegression()
    lr.fit(X, y)

    val = np.array(val).reshape(1,-1)

    return lr.predict(val)[0]