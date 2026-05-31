import numpy as np


class Value:
    """
    Armazena seu valor, seu gradiente e função para calcular o gradiente do valor
    em relação a ele mesmo e seus filhos"""
    def __init__(self, data: np.ndarray, _children=(), _op=''):
        self.data = data
        self.grad = np.zeros_like(data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            if self.grad.shape == out.grad.shape:
                self.grad += out.grad
            else:
                self.grad += np.sum(out.grad, axis=0, keepdims=True)

            if other.grad.shape == out.grad.shape:
                other.grad += out.grad
            else:
                other.grad += np.sum(out.grad, axis=0, keepdims=True)
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= self.data * out.grad / (other.data ** 2)

        out._backward = _backward
        return out

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data.shape}, grad={self.grad.shape})"
    
    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self.data.shape


def relu(value: Value):
    # Usa np.maximum para aplicar ReLU em todo o array
    out = Value(np.maximum(0, value.data), (value,), 'ReLU')

    def _backward():
        # O gradiente é 1 onde out.data > 0, senão 0
        value.grad += (out.data > 0) * out.grad

    out._backward = _backward

    return out

def sigmoid(value: Value):
    # função sigmoide: 1 / (1 + e^(-x)), vetorizada
    s = 1 / (1 + np.exp(-value.data))
    out = Value(s, (value,), 'Sigmoid')

    def _backward():
        # derivada da sigmoide: s * (1 - s), também vetorizada
        value.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out


# Funções de perda
def binary_cross_entropy(y_true: np.ndarray, y_pred: Value):
    eps = 1e-12
    y_true = y_true.reshape(-1, 1)   # garante shape (N,1)
    y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred_clipped) +
        (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    out = Value(loss, (y_pred,), 'BCE')

    def _backward():
        # derivada em relação a y_pred
        grad_pred = (-(y_true / y_pred_clipped) +
                     ((1 - y_true) / (1 - y_pred_clipped))) / y_true.size
        y_pred.grad += grad_pred * out.grad

    out._backward = _backward
    return out


def mse(y_true: np.ndarray, y_pred: Value):
    # força y_true a ter shape (N,1)
    y_true = y_true.reshape(-1, 1)
    diff = y_true - y_pred.data
    loss = np.mean(diff ** 2)

    out = Value(loss, (y_pred,), 'MSE')

    def _backward():
        grad_pred = (-2 * diff) / y_true.size
        # grad_pred tem shape (N,1), compatível com y_pred.grad
        y_pred.grad += grad_pred * out.grad

    out._backward = _backward
    return out


def r2_loss(y_true: np.ndarray, y_pred: Value):
    y_true = y_true.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred.data) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    loss = ss_res / ss_tot  # equivalente a (1 - R²)

    out = Value(loss, (y_pred,), 'R2Loss')

    def _backward():
        grad_pred = -2 * (y_true - y_pred.data) / ss_tot
        y_pred.grad += grad_pred * out.grad

    out._backward = _backward
    return out


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class LinearLayer(Module):
    def __init__(self, input_shape, output_shape, activation=relu):
        limit = np.sqrt(6 / (input_shape + output_shape))
        self.w = Value(np.random.uniform(-limit, limit, size=(input_shape, output_shape)), _op=f'linW')
        self.b = Value(np.zeros((1, output_shape)), _op=f'linb')
        self.activation = activation

    def __call__(self, x: np.ndarray):
        inputs = Value(x) if not isinstance(x, Value) else x
        out = inputs @ self.w + self.b
        return self.activation(out)

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"LinearLayer(w={self.w.shape}, b={self.b.shape}), activation={self.activation.__name__})"


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p
                for layer in self.layers
                for p in layer.parameters()
        ]

    def __repr__(self):
        return f"Sequential of [{',\n\t'.join(str(layer) for layer in self.layers)}\n]"


class MLP(Sequential):
    def __init__(
            self, input_size, hidden_sizes, output_size,
            loss_function,
            activation=relu, output_activation=lambda x: x,
            epochs=100, lr=0.01, decay=0.5,
            tolerance=1e-4, patience=5, clip_grad=False
            ):
        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                LinearLayer(
                    sizes[i],
                    sizes[i+1],
                    activation if i < len(sizes) - 2 else output_activation)
            )
        super().__init__(*layers)
        self.loss_function = loss_function
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.tolerance = tolerance
        self.patience = patience
        self.clip_grad = clip_grad

    def fit(self, X, y):
        best_loss = float("inf")
        lr = self.lr
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self(X)
            # Compute loss
            loss = self.loss_function(y, y_pred)
            current_loss = loss.data
            # Backward pass
            self.zero_grad()
            loss.backward()
            # Update parameters
            for p in self.parameters():
                if self.clip_grad:
                    p.grad = np.clip(p.grad, -5, 5)

                p.data -= self.lr * p.grad

            # Checa melhora da loss
            if best_loss - current_loss > self.tolerance:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Se não melhorar por "patience" épocas, reduz lr
            if epochs_no_improve >= self.patience:
                lr *= self.decay
                epochs_no_improve = 0

            # Mostrar loss e lr na mesma linha
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.data:.6f} - LR: {lr:.6f}", end="\r")

        print()


    def predict(self, X):
        return self(X).data

    def __repr__(self):
        return f"MLP of [{',\n\t'.join(str(layer) for layer in self.layers)}\n]"
