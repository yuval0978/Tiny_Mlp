
# ------------------ PID and TinyMLP ------------------
class PIDController:
    def __init__(self, Kp=12.0, Ki=0.8, Kd=0.25, integral_limit=100.0):
        self.Kp = float(Kp); self.Ki = float(Ki); self.Kd = float(Kd)
        self.integral = 0.0; self.last_e = 0.0
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0; self.last_e = 0.0

    def compute(self, e, dt):
        self.integral += e * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        deriv = (e - self.last_e) / (dt if dt > 0 else 1e-8)
        self.last_e = e
        u = self.Kp * e + self.Ki * self.integral + self.Kd * deriv
        return u


class TinyMLP:
    def __init__(self, input_dim=10, hidden=48, out_dim=3, seed=0):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(hidden, input_dim) * 0.1
        self.b1 = np.zeros((hidden,))
        self.w2 = rng.randn(out_dim, hidden) * 0.1
        self.b2 = np.zeros((out_dim,))

    def forward(self, x):
        z1 = self.w1.dot(x) + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = self.w2.dot(a1) + self.b2
        self._x, self._z1, self._a1, self._z2 = x, z1, a1, z2
        return z2.copy()

    def backward_from_dout(self, dL_dout):
        self.dw2 = np.outer(dL_dout, self._a1)
        self.db2 = dL_dout.copy()
        da1 = self.w2.T.dot(dL_dout)
        dz1 = da1 * (self._z1 > 0).astype(float)
        self.dw1 = np.outer(dz1, self._x)
        self.db1 = dz1.copy()

    def step_sgd(self, lr=1e-4, clip=1.0):
        for g in [self.dw1, self.dw2, self.db1, self.db2]:
            norm = np.linalg.norm(g)
            if norm > clip and norm > 0:
                g *= (clip / norm)
        self.w1 -= lr * self.dw1; self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2; self.b2 -= lr * self.db2

def build_feature_vector(e_hist, integral, u_hist, sp, noise_est=0.0, env_flag=0.0):
    return np.array([
        e_hist[0], e_hist[1], e_hist[2],
        integral, (e_hist[0] - e_hist[1]),
        u_hist[0], u_hist[1],
        sp, noise_est, env_flag
    ], dtype=np.float32)


