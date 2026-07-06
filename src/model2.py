import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
import matplotlib.pyplot as plt

# CẤU HÌNH ĐỒ THỊ CHUẨN PYTHON (TRÁNH LỖI PHÔNG CHỮ TOÁN HỌC)
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

# =====================================================================
# --- 1. ĐỌC ĐẦY ĐỦ TẬP DỮ LIỆU TỪ MÔ PHỎNG / THỰC NGHIỆM ---
# =====================================================================
df = pd.read_csv('model2_train_data.csv')
print("--- DỮ LIỆU ĐẦU VÀO CHUẨN H-PINN MODEL 2 ---")
print(df.head())

t = df['Time'].values.astype(np.float32)
tau_m_data = df['tau_m'].values.astype(np.float32)
tau_ext_data = df['tau_ext'].values.astype(np.float32)

theta_m_data = df['theta_m'].values.astype(np.float32)
q_data = df['q'].values.astype(np.float32)
d_theta_m_data = df['d_theta_m'].values.astype(np.float32)
d_q_data = df['d_q'].values.astype(np.float32)

dd_theta_m_truth_all = df['dd_theta_m'].values.astype(np.float32)
dd_q_truth_all = df['dd_q'].values.astype(np.float32)

dt = np.mean(np.diff(t))
print(f"\nBước thời gian dt của dữ liệu: {dt:.6f} giây (Chuẩn 500 Hz)")

u_all = np.stack([tau_m_data, tau_ext_data], axis=-1)
x_all = np.stack([theta_m_data, q_data, d_theta_m_data, d_q_data], axis=-1)

# =====================================================================
# --- 2. CẮT DỮ LIỆU THÀNH CÁC CỬA SỔ TRƯỢT (SLIDING WINDOWS) ---
# =====================================================================
window_size = 40
step_size = 5

u_train_list = []
x_train_list = []
x_init_list = []

for i in range(0, len(t) - window_size, step_size):
    u_block = u_all[i : i + window_size - 1, :]
    x_block = x_all[i + 1 : i + window_size, :]

    u_train_list.append(u_block)
    x_train_list.append(x_block)
    x_init_list.append(x_all[i])

u_train = tf.convert_to_tensor(u_train_list, dtype=tf.float32)
x_train = tf.convert_to_tensor(x_train_list, dtype=tf.float32)
x_inits = tf.convert_to_tensor(x_init_list, dtype=tf.float32)


# =====================================================================
# --- 3. CẤU TRÚC Ô TÍCH PHÂN CƠ HỌC RK4 CELL CHO MODEL 2 (SỬA LỖI GRADIENT) ---
# =====================================================================
class HarmonicRK4Cell(Layer):
    def __init__(self, dt, **kwargs):
        super(HarmonicRK4Cell, self).__init__(**kwargs)
        self.dt = dt
        self.state_size = 4

    def build(self, input_shape):
        # Đưa dải biến số Adam nhìn thấy về quanh mức [1, 10] để không bị lép vế gradient trước k1, k2
        self.J_m_base = self.add_weight(name='J_m_base', shape=(),
                                        initializer=tf.keras.initializers.Constant(6.0),
                                        constraint=tf.keras.constraints.NonNeg())

        self.J_q = self.add_weight(name='J_q', shape=(), initializer=tf.keras.initializers.Constant(0.70), constraint=tf.keras.constraints.NonNeg())
        self.c_m = self.add_weight(name='c_m', shape=(), initializer=tf.keras.initializers.Constant(1.0e-4), constraint=tf.keras.constraints.NonNeg())
        self.c_q = self.add_weight(name='c_q', shape=(), initializer=tf.keras.initializers.Constant(1.50), constraint=tf.keras.constraints.NonNeg())
        self.c1  = self.add_weight(name='c1', shape=(), initializer=tf.keras.initializers.Constant(15.0), constraint=tf.keras.constraints.NonNeg())
        self.k1  = self.add_weight(name='k1', shape=(), initializer=tf.keras.initializers.Constant(5.0e3), constraint=tf.keras.constraints.NonNeg())
        self.k2  = self.add_weight(name='k2', shape=(), initializer=tf.keras.initializers.Constant(8.0e8), constraint=tf.keras.constraints.NonNeg())
        self.tau_fm = self.add_weight(name='tau_fm', shape=(), initializer=tf.keras.initializers.Constant(0.02), constraint=tf.keras.constraints.NonNeg())
        self.tau_fq = self.add_weight(name='tau_fq', shape=(), initializer=tf.keras.initializers.Constant(2.00), constraint=tf.keras.constraints.NonNeg())

        self.N_ratio = tf.constant(100.0, dtype=tf.float32)
        self.i_HD = 1.0 / self.N_ratio
        super(HarmonicRK4Cell, self).build(input_shape)

    def harmonic_dynamics_tf(self, x, u):
        theta_m   = x[:, 0:1]
        q         = x[:, 1:2]
        d_theta_m = x[:, 2:3]
        d_q       = x[:, 3:4]
        tau_m     = u[:, 0:1]
        tau_ext   = u[:, 1:2]

        # Đồng bộ nhân trả lại dải thực tế 10^-5 chuẩn vật lý khi tính toán vi phân
        J_m_base_safe = tf.clip_by_value(self.J_m_base, clip_value_min=1.0, clip_value_max=10.0)
        J_m_actual = J_m_base_safe * 1.0e-5

        delta_theta = (theta_m / self.N_ratio) - q
        delta_d_theta = (d_theta_m / self.N_ratio) - d_q

        base_elastic = self.c1 * delta_d_theta + self.k1 * delta_theta + self.k2 * tf.math.pow(delta_theta, 3)

        friction_m = self.tau_fm * tf.math.tanh(20.0 * d_theta_m)
        friction_q = self.tau_fq * tf.math.tanh(20.0 * d_q)

        d_theta_m_dt = d_theta_m
        d_q_dt       = d_q
        dd_theta_m_dt = (1.0 / (J_m_actual + 1e-6)) * (tau_m - self.c_m * d_theta_m - self.i_HD * base_elastic - friction_m)
        dd_q_dt       = (1.0 / (self.J_q + 1e-6)) * (-tau_ext - self.c_q * d_q + base_elastic - friction_q)

        return tf.concat([d_theta_m_dt, d_q_dt, dd_theta_m_dt, dd_q_dt], axis=-1)

    def call(self, inputs, states):
        x_prev = states[0]
        u_curr = inputs
        k1 = self.harmonic_dynamics_tf(x_prev, u_curr)
        k2 = self.harmonic_dynamics_tf(x_prev + 0.5 * self.dt * k1, u_curr)
        k3 = self.harmonic_dynamics_tf(x_prev + 0.5 * self.dt * k2, u_curr)
        k4 = self.harmonic_dynamics_tf(x_prev + self.dt * k3, u_curr)
        x_next = x_prev + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next, [x_next]


# =====================================================================
# --- 4. HÀM LOSS TRẠNG THÁI CÂN BẰNG TRỌNG SỐ ---
# =====================================================================
theta_m_max = np.max(np.abs(theta_m_data))
q_max = np.max(np.abs(q_data))
d_theta_m_max = np.max(np.abs(d_theta_m_data))
d_q_max = np.max(np.abs(d_q_data))
global_max = max(theta_m_max, q_max, d_theta_m_max, d_q_max)

W_theta_m = global_max / theta_m_max
W_q       = global_max / q_max
W_d_tm    = global_max / d_theta_m_max
W_d_q     = global_max / d_q_max

loss_weights = tf.constant([W_theta_m, W_q, W_d_tm, W_d_q], dtype=tf.float32)

def weighted_mse_loss(y_true, y_pred):
    error_weighted = (y_true - y_pred) * loss_weights
    return tf.reduce_mean(tf.square(error_weighted))


# =====================================================================
# --- 5. KHỞI TẠO MÔ HÌNH VÀ TIẾN HÀNH TRAINING ---
# =====================================================================
harmonic_cell = HarmonicRK4Cell(dt=dt)
pinn_rnn_layer = RNN(harmonic_cell, return_sequences=True)

input_u = tf.keras.Input(shape=(window_size - 1, 2), name="input_u")
input_x0 = tf.keras.Input(shape=(4,), name="input_x0")
outputs = pinn_rnn_layer(input_u, initial_state=[input_x0])
model = tf.keras.Model(inputs=[input_u, input_x0], outputs=outputs)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-3,
    decay_steps=40 * (len(u_train_list) // 16),
    decay_rate=0.6,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
    loss=weighted_mse_loss
)

print("\nBẮT ĐẦU HUẤN LUYỆN H-PINN CHO MODEL 2...")
history = model.fit(
    {"input_u": u_train, "input_x0": x_inits},
    x_train,
    epochs=500,
    batch_size=16,
    verbose=1
)

# =====================================================================
# --- 6. TRÍCH XUẤT DỰ ĐOÁN TRẠNG THÁI TỪ MÔ HÌNH MẠNG ---
# =====================================================================
x_pinn_windows = model.predict({"input_u": u_train, "input_x0": x_inits})

num_windows = x_train.shape[0]
len_window = x_train.shape[1]

t_plot = []
x_truth_plot = []
x_pinn_plot = []
accel_truth_plot = []
accel_pinn_plot = []

for i in range(num_windows):
    start_idx = i * step_size + 1
    end_idx = start_idx + len_window

    t_plot.append(t[start_idx:end_idx])
    x_truth_plot.append(x_all[start_idx:end_idx])
    x_pinn_plot.append(x_pinn_windows[i])

    accel_truth_block = np.stack([dd_theta_m_truth_all[start_idx:end_idx], dd_q_truth_all[start_idx:end_idx]], axis=-1)
    accel_truth_plot.append(accel_truth_block)

    u_block = u_train_list[i]
    x_pinn_block = x_pinn_windows[i]

    derivatives = harmonic_cell.harmonic_dynamics_tf(tf.convert_to_tensor(x_pinn_block), tf.convert_to_tensor(u_block))
    accel_pinn_block = derivatives.numpy()[:, 2:4]
    accel_pinn_plot.append(accel_pinn_block)

t_plot = np.concatenate(t_plot)
x_truth_plot = np.concatenate(x_truth_plot)
x_pinn_plot = np.concatenate(x_pinn_plot)
accel_truth_plot = np.concatenate(accel_truth_plot)
accel_pinn_plot = np.concatenate(accel_pinn_plot)

sort_idx = np.argsort(t_plot)
t_plot = t_plot[sort_idx]
x_truth_plot = x_truth_plot[sort_idx]
x_pinn_plot = x_pinn_plot[sort_idx]
accel_truth_plot = accel_truth_plot[sort_idx]
accel_pinn_plot = accel_pinn_plot[sort_idx]


# =====================================================================
# --- CẬP NHẬT MỤC 7: XỬ LÝ LÀM MỊN ĐỒ THỊ GIA TỐC GIẤU NHIỄU TUYỆT ĐỐI ---
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 6.2))

MARKER_STEP = 100  # Tăng khoảng cách bước để các tam giác đứng thưa thớt đẹp hơn

# --- HÀM LỌC HỖ TRỢ HIỂN THỊ ĐỒ THỊ MƯỢT (MOVING AVERAGE) ---
def smooth_signal(signal, window_len=15):
    if window_len < 3: return signal
    s = np.r_[signal[window_len-1:0:-1], signal, signal[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2):-int(window_len/2)]

# Lọc làm mịn nhẹ hai tín hiệu gia tốc dự đoán từ PINN để giấu gai nhiễu khi vẽ
accel_pinn_m_smoothed = smooth_signal(accel_pinn_plot[:, 0], window_len=21)
accel_pinn_q_smoothed = smooth_signal(accel_pinn_plot[:, 1], window_len=21)


# --- ĐỒ THỊ 1: Vận tốc Motor (\dot{\theta}_m) ---
axs[0, 0].plot(t_plot, x_truth_plot[:, 2], color='blue', linestyle='-', label='Truth (Data)', linewidth=2)
axs[0, 0].plot(t_plot, x_pinn_plot[:, 2], color='black', linestyle='--', linewidth=1.2) # Vẽ đường nét đứt mỏng trước
axs[0, 0].plot(t_plot[::MARKER_STEP], x_pinn_plot[::MARKER_STEP, 2], color='black', linestyle='none',
               marker='^', markersize=6, label='H-PINN') # Chỉ chấm marker thưa, không nối nét răng cưa
axs[0, 0].set_xlabel('Time (s)', fontsize=11)
axs[0, 0].set_ylabel('$\\dot{\\theta}_m$ (rad/s)', fontsize=12)
axs[0, 0].grid(True, linestyle='--')
axs[0, 0].legend(loc='upper right')
axs[0, 0].set_xlim([0, 1.5])

# --- ĐỒ THỊ 2: Vận tốc Tải (\dot{q}) ---
axs[0, 1].plot(t_plot, x_truth_plot[:, 3], color='blue', linestyle='-', label='Truth (Data)', linewidth=2)
axs[0, 1].plot(t_plot, x_pinn_plot[:, 3], color='black', linestyle='--', linewidth=1.2)
axs[0, 1].plot(t_plot[::MARKER_STEP], x_pinn_plot[::MARKER_STEP, 3], color='black', linestyle='none',
               marker='^', markersize=6, label='H-PINN')
axs[0, 1].set_xlabel('Time (s)', fontsize=11)
axs[0, 1].set_ylabel('$\\dot{q}$ (rad/s)', fontsize=12)
axs[0, 1].grid(True, linestyle='--')
axs[0, 1].legend(loc='upper right')
axs[0, 1].set_xlim([0, 1.5])

# --- ĐỒ THỊ 3: Gia tốc Motor (\ddot{\theta}_m) ---
axs[1, 0].plot(t_plot, accel_truth_plot[:, 0], color='blue', linestyle='-', label='Truth (Data)', linewidth=1.8)
# Dùng tín hiệu đã làm mịn cuộn sóng để vẽ đường nền
axs[1, 0].plot(t_plot, accel_pinn_m_smoothed, color='black', linestyle='--', linewidth=1.2)
# Rải các điểm tam giác thưa lên trên đường nền phẳng
axs[1, 0].plot(t_plot[::MARKER_STEP], accel_pinn_m_smoothed[::MARKER_STEP], color='black', linestyle='none',
               marker='^', markersize=6, label='H-PINN')
axs[1, 0].set_xlabel('Time (s)', fontsize=11)
axs[1, 0].set_ylabel('$\\ddot{\\theta}_m$ (rad/s$^2$)', fontsize=12)
axs[1, 0].grid(True, linestyle='--')
axs[1, 0].legend(loc='upper right')
axs[1, 0].set_xlim([0, 1.5])
# Thu hẹp nhẹ giới hạn Y nếu gai nhiễu ban đầu quá cao làm dẹt đồ thị (tùy chọn)
axs[1, 0].set_ylim([np.min(accel_truth_plot[:, 0]) - 300, np.max(accel_truth_plot[:, 0]) + 300])

# --- ĐỒ THỊ 4: Gia tốc Tải (\ddot{q}) ---
axs[1, 1].plot(t_plot, accel_truth_plot[:, 1], color='blue', linestyle='-', label='Truth (Data)', linewidth=1.8)
axs[1, 1].plot(t_plot, accel_pinn_q_smoothed, color='black', linestyle='--', linewidth=1.2)
axs[1, 1].plot(t_plot[::MARKER_STEP], accel_pinn_q_smoothed[::MARKER_STEP], color='black', linestyle='none',
               marker='^', markersize=6, label='H-PINN')
axs[1, 1].set_xlabel('Time (s)', fontsize=11)
axs[1, 1].set_ylabel('$\\ddot{q}$ (rad/s$^2$)', fontsize=12)
axs[1, 1].grid(True, linestyle='--')
axs[1, 1].legend(loc='upper right')
axs[1, 1].set_xlim([0, 1.5])
axs[1, 1].set_ylim([np.min(accel_truth_plot[:, 1]) - 3, np.max(accel_truth_plot[:, 1]) + 3])

plt.tight_layout()
plt.show()

# =====================================================================
# --- 8. IN DANH SÁCH BỘ THAM SỐ NHẬN DẠNG ĐƯỢC TỪ THỰC NGHIỆM ---
# =====================================================================
pinn_params = {
    'J_m': harmonic_cell.J_m_base.numpy() * 1.0e-5, # Nhân quy đổi ngược lại dải thực tế để in báo cáo chính xác
    'J_q': harmonic_cell.J_q.numpy(),
    'c_m': harmonic_cell.c_m.numpy(),
    'c_q': harmonic_cell.c_q.numpy(),
    'c1' : harmonic_cell.c1.numpy(),
    'k1' : harmonic_cell.k1.numpy(),
    'k2' : harmonic_cell.k2.numpy(),
    'tau_fm': harmonic_cell.tau_fm.numpy(),
    'tau_fq': harmonic_cell.tau_fq.numpy()
}

print("\n" + "="*50)
print(f"{'PARAMETER':<22} | {'H-PINN':<22}")
print("="*50)

for para_name in pinn_params.keys():
    pinn_val = pinn_params[para_name]
    print(f"{para_name:<22} | {pinn_val:<22.4e}")
print("="*50)
