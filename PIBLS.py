import numpy as np
from scipy.linalg import pinv


def exact_solution(x, y):
    """TC-5精确解: u = 1/2 + exp(-(2x^2 + 4y^2))"""
    return 0.5 + np.exp(-(2 * x ** 2 + 4 * y ** 2))


def source(x, y):
    """TC-5源项: R = u_xx + u_yy"""
    exp_term = np.exp(-(2 * x ** 2 + 4 * y ** 2))
    u_xx = exp_term * (16 * x ** 2 - 4)
    u_yy = exp_term * (64 * y ** 2 - 8)
    return u_xx + u_yy


class PIBLS:
    """物理信息宽度学习系统（扩散方程版本）"""

    def __init__(self, N1, N2, map_func='tanh', enhance_func='sigmoid'):
        self.N1 = int(N1)
        self.N2 = int(N2)

        self.map_act_name, self.map_activation = self._get_activation(map_func)
        self.enhance_act_name, self.enhance_activation = self._get_activation(enhance_func)

        self.map_derivative = self._get_derivative(map_func)
        self.map_second_derivative = self._get_second_derivative(map_func)
        self.enhance_derivative = self._get_derivative(enhance_func)
        self.enhance_second_derivative = self._get_second_derivative(enhance_func)

        self.W_map = None
        self.B_map = np.random.randn(self.N1)
        self.W_enhance = np.random.randn(self.N1, self.N2)
        self.B_enhance = np.random.randn(self.N2)

        self.beta = None
        self.is_initialized = False

    def _get_activation(self, activation):
        activations = {
            'relu': ('relu', lambda x: np.maximum(0, x)),
            'tanh': ('tanh', lambda x: np.tanh(x)),
            'sigmoid': ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
            'linear': ('linear', lambda x: x)
        }
        return activations.get(activation.lower(), activations['tanh'])

    def _get_derivative(self, activation):
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'linear': lambda x: np.ones_like(x)
        }
        return derivatives.get(activation.lower(), derivatives['tanh'])

    def _get_second_derivative(self, activation):
        second_derivatives = {
            'tanh': lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2),
            'relu': lambda x: np.zeros_like(x),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))) * (1 - 2 / (1 + np.exp(-x))),
            'linear': lambda x: np.zeros_like(x)
        }
        return second_derivatives.get(activation.lower(), second_derivatives['tanh'])

    def _build_features(self, x, y):
        X_bias = np.column_stack([x, y, np.ones_like(x)])
        if not self.is_initialized:
            self._initialize_weights(X_bias)

        Z_map = X_bias @ self.W_map + self.B_map
        H_map = self.map_activation(Z_map)

        Z_enhance = H_map @ self.W_enhance + self.B_enhance
        H_enhance = self.enhance_activation(Z_enhance)

        return np.hstack([H_map, H_enhance]), (Z_map, Z_enhance)

    def _initialize_weights(self, X_bias):
        init_W = np.random.randn(3, self.N1)
        self.W_map = self.sparse_bls(X_bias, X_bias @ init_W)
        self.is_initialized = True

    def shrinkage(self, a, b):
        return np.sign(a) * np.maximum(np.abs(a) - b, 0)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = A.T.dot(A)
        m = A.shape[1]
        n = b.shape[1]
        x1 = np.zeros([m, n])
        wk = ok = uk = x1
        L1 = np.linalg.inv(AA + np.eye(m))
        L2 = L1.dot(A.T).dot(b)
        for _ in range(itrs):
            ck = L2 + L1.dot(ok - uk)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def _compute_derivatives(self, x, y, z_values):
        Z_map, Z_enhance = z_values

        dH_map = self.map_derivative(Z_map)
        ddH_map = self.map_second_derivative(Z_map)

        dH_dx_map = dH_map * self.W_map[0, :]
        dH_dy_map = dH_map * self.W_map[1, :]

        d2H_dx2_map = ddH_map * (self.W_map[0, :] ** 2)
        d2H_dy2_map = ddH_map * (self.W_map[1, :] ** 2)

        dH_enhance = self.enhance_derivative(Z_enhance)
        ddH_enhance = self.enhance_second_derivative(Z_enhance)

        d2H_dx2_enhance = ddH_enhance * (dH_dx_map @ self.W_enhance) ** 2 + \
                          dH_enhance * (d2H_dx2_map @ self.W_enhance)

        d2H_dy2_enhance = ddH_enhance * (dH_dy_map @ self.W_enhance) ** 2 + \
                          dH_enhance * (d2H_dy2_map @ self.W_enhance)

        d2H_dx2 = np.hstack([d2H_dx2_map, d2H_dx2_enhance])
        d2H_dy2 = np.hstack([d2H_dy2_map, d2H_dy2_enhance])

        return d2H_dx2, d2H_dy2

    def build_system(self, pde_data, bc_data):
        x_pde, y_pde = pde_data
        x_bc, y_bc = bc_data

        H_pde, z_pde = self._build_features(x_pde, y_pde)
        d2H_dx2, d2H_dy2 = self._compute_derivatives(x_pde, y_pde, z_pde)

        A_pde = d2H_dx2 + d2H_dy2
        b_pde = source(x_pde, y_pde)

        H_bc, _ = self._build_features(x_bc, y_bc)
        b_bc = exact_solution(x_bc, y_bc)

        A_matrix = np.vstack([A_pde, H_bc])
        b_vector = np.concatenate([b_pde, b_bc])

        return A_matrix, b_vector

    def fit(self, pde_data, bc_data):
        A, b = self.build_system(pde_data, bc_data)
        self.beta = pinv(A) @ b.reshape(-1, 1)
        return self.beta

    def predict(self, x, y):
        if self.beta is None:
            raise ValueError("Model not trained. Call fit() first.")
        H, _ = self._build_features(x, y)
        return (H @ self.beta).flatten()
