import numpy as np 
from scipy.optimize import minimize 

class LogReg:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.N = len(y)
        self.classes = np.unique(self.y)
        
        def NLL(beta):
            ones = np.ones((self.N, 1)) 
            X_ = np.hstack((ones, self.X))
            z = np.dot(X_, beta)
            z = z.reshape(-1,)
            p = (1 / (1 + np.exp(z))) 
            pi = np.where(self.y==self.classes[1], 1-p, p)
            nll = np.sum(np.log(pi))*(-1)
            return nll 
        
        
        beta_guess = np.zeros(self.X.shape[1] + 1)
        min_results = minimize(NLL, beta_guess)
        self.coefficients = min_results.x
        self.loss = round(NLL(self.coefficients),4)
        self.accuracy = round(self.score(X, y),4)
    
    def predict_proba(self, X):
        self.X = np.array(X)
        ones = np.ones((self.X.shape[0], 1))
        X_ = np.hstack((ones, X))
        z = np.dot(X_, self.coefficients)
        z = z.reshape(-1,)
        p = (1 / (1 + np.exp(-z))) 
        
        return p
    
    def predict(self, X, t=0.5):
        self.X = np.array(X)
        out = np.where(self.predict_proba(X)<t, self.classes[0], self.classes[1])
        return out
    
    def score(self, X, y, t=0.5):
        self.X = np.array(X)
        self.y = np.array(y)
        
        
        score = np.sum(self.predict(X,t) == self.y) / len(self.y)
        return score
    
    def summary(self):
        print('+-------------------------------+')
        print('|  Logistic Regression Summary  |')
        print('+-------------------------------+')
        
        print('Number of training observations: ' + str(self.N))
        print('Coefficient Estimated: ' + str(self.coefficients))
        print('Log-likelihood: ' + str(round(self.loss, 4)))
        print('Accuracy: ' + str(self.accuracy))
        
    
    
    def precision_recall(self, X, y, t = 0.5):
        self.X = np.array(X)
        self.y = np.array(y)
        
        
        
        pr0 =  round(np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[0])) /
                    (np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[0])) +
        np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[1]))), 4)
    
        rcl0 = round(np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[0])) /
                       np.sum(self.y==self.classes[0]), 4)
        
        
        pr1 =  round(np.sum((self.predict(X,t) == self.classes[1]) &
                            (self.y == self.classes[1])) / 
                    (np.sum((self.predict(X,t) == self.classes[1]) &
                            (self.y == self.classes[1])) + 
        np.sum((self.predict(X,t) == self.classes[1]) &
                             (self.y == self.classes[0]))), 4)
        
        rcl1 = round(np.sum((self.predict(X,t) == self.classes[1]) &
                            (self.y == self.classes[1]))/ 
                      np.sum(self.y==self.classes[1]), 4)
        
        
        
        print('Class: ' + str(self.classes[0]))
        print('  Precision = ' + str(pr0))
        print('  Recall    = ' + str(rcl0))
        
        print('Class: ' + str(self.classes[1]))
        print('  Precision = ' + str(pr1))
        print('  Recall    = ' + str(rcl1))

