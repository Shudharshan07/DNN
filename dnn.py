import numpy as np
import pandas as pd




# y(b, out) = x(b, In) * W(IN, out) + B(b, out)

# dy3/dW3 = x2 -- the x we stored
# dy/dB = 1

# dy2/dw2 = x2 , y2 = dy3/dw3 * w2(IN, Out) 


class Data:
    def __init__(self, x, y):
        prem = np.random.permutation(len(x))
        self.X = np.array(x)
        self.Y = np.array(y)

        self.X = self.X[prem]
        self.Y = self.Y[prem]

        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)

        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)
        
    def __iter__(self):
        return iter(zip(self.X, self.Y))
        
        
class MSE:
    def __call__(self,Y_a: np.ndarray, Y_p: np.ndarray):
        return np.mean((Y_p - Y_a) ** 2)
    

    def _derivative(self,Y_a: np.ndarray, Y_p: np.ndarray):
        return (2 * (Y_p - Y_a)) / Y_a.size
        

class Sequential:
    def __init__(self, *args, **kargs):
        self._layers = args
        self._batch_size = 1
        self.lr = 0.01
        self.loss = MSE()
        if "batch_size" in kargs:
            self._batch_size = kargs["batch_size"]
        
        if "lr" in kargs:
            self.lr = kargs["lr"]

        for layer in self._layers:
            layer.lr = self.lr

        self._Data = None

    def fit(self, df_X: pd.DataFrame, df_Y: pd.DataFrame):
        self._Data = Data(df_X, df_Y)

        
    def _back(self, pred, actual):
        grad = self.loss._derivative(actual, pred)
        for layer in reversed(self._layers):
            grad = layer._backward(grad)
        

    def train(self, epoch = 1):
        for e in range(epoch):
            loss_v = 0.0
            for x, y in self._Data:
                for layer in self._layers:
                    x = layer._forward(x)

                loss_v += self.loss(x, y)
                self._back(x, y)

            print(loss_v)
                    
class Layer:
    def __init__(self, IN: int, OUT: int):        
        self.W = np.random.rand(IN, OUT)
        self.B = np.random.rand(1, OUT)
        self.X = None
        self.lr = 0.01

    def _backward(self, grad):
        prev = np.dot(grad, self.W.T)
        self.W -= self.lr * (np.dot(self.X.T, grad))
        self.B -= self.lr * grad

        return prev
        

    def _forward(self, X_IN: np.ndarray):
        self.X = X_IN.reshape(1, -1)
        return np.dot(X_IN, self.W) + self.B




df = pd.read_csv("train.csv")

a = Sequential(Layer(1,8), Layer(8,1), batch_size = 2)
a.fit(df["in"], df["out"])

a.train(30)
