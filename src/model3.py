import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
import matplotlib.pyplot as plt
import math

# =====================================================================
# 1. ĐỌC DỮ LIỆU TỪ FILE CSV ĐƯỢC XUẤT TỪ MATLAB
# =====================================================================
df = pd.read_csv('model3_simulation.csv')
print("--- 5 dòng đầu của dữ liệu CSV ---")
print(df.head())

t          = df['Time'].values.astype(np.float32)
i_data     = df['Current'].values.astype(np.float32)
theta_m    = df['Theta_m'].values.astype(np.float32)
q_load     = df['Q_load'].values.astype(np.float32)
dtheta_m   = df['Speed'].values.astype(np.float32)       # dot_theta_m
dq_data    = df['DotQ'].values.astype(np.float32)

dt = np.mean(np.diff(t))
print(f"\nBước thời gian dt: {dt:.6f} giây")

# =====================================================================
# 2. CHUẨN BỊ DỮ LIỆU DẠNG CỬA SỔ TRƯỢT (WINDOWING) CHO RNN
# =====================================================================
Vs_data      = 12.0 * np.ones_like(t)
tau_ext_data =  0.0 * np.ones_like(t)

u_all = np.stack([Vs_data, tau_ext_data], axis=-1)
x_all = np.stack([i_data, theta_m, q_load, dtheta_m, dq_data], axis=-1)

window_size = 40
step_size   = 5

u_train_list = []
x_train_list = []
x_init_list  = []

for i in range(0, len(t) - window_size, step_size):
    u_train_list.append(u_all[i : i + window_size - 1, :])
    x_train_list.append(x_all[i + 1 : i + window_size, :])
    x_init_list.append(x_all[i])

u_train = tf.convert_to_tensor(u_train_list, dtype=tf.float32)
x_train = tf.convert_to_tensor(x_train_list, dtype=tf.float32)
x_inits = tf.convert_to_tensor(x_init_list,  dtype=tf.float32)

print(f"u_train shape : {u_train.shape}")
print(f"x_train shape : {x_train.shape}")
print(f"x_inits shape : {x_inits.shape}")

# =====================================================================
# 3. ĐỊNH NGHĨA LỚP RK4 CELL — TÁCH BIỆT THAM SỐ cm VÀ cb
# =====================================================================
class Model3RK4Cell(Layer):
    def __init__(self, dt, **kwargs):
        super().__init__(**kwargs)
        self.dt         = dt
        self.state_size = 5  # [i, theta_m, q, dot_theta_m, dot_q]

        # Hằng số vật lý cố định cấu hình từ MATLAB
        self.L     = tf.constant(0.025,       dtype=tf.float32)
        self.J_sum = tf.constant(7.135e-4,    dtype=tf.float32)  # Jr + Jm = 0.00065 + 6.5e-5
        self.Jq    = tf.constant(2.7780,      dtype=tf.float32)  # Đã đồng nhất chính xác 2.7780
        self.N     = tf.constant(101.0,       dtype=tf.float32)
        self.iHD   = tf.constant(1.0/101.0,   dtype=tf.float32)

    def build(self, input_shape):
        # Không gian Log-space khởi tạo các tham số cần nhận dạng (Đoán mò ban đầu)
        self.log_R      = self.add_weight(name='log_R',
                            shape=(), initializer=tf.constant_initializer(math.log(1.15)))
        self.log_kb     = self.add_weight(name='log_kb',
                            shape=(), initializer=tf.constant_initializer(math.log(0.095)))
        self.log_kt     = self.add_weight(name='log_kt',
                            shape=(), initializer=tf.constant_initializer(math.log(0.045)))

        # TÁCH RIÊNG KHỞI TẠO cm VÀ cb THAY VÌ ĐỂ CHUNG C_sum
        self.log_cm     = self.add_weight(name='log_cm',
                            shape=(), initializer=tf.constant_initializer(math.log(3.0e-2)))
        self.log_cb     = self.add_weight(name='log_cb',
                            shape=(), initializer=tf.constant_initializer(math.log(1.4e-4)))

        self.log_tau_fm = self.add_weight(name='log_tau_fm',
                            shape=(), initializer=tf.constant_initializer(math.log(0.05)))
        self.log_cq     = self.add_weight(name='log_cq',
                            shape=(), initializer=tf.constant_initializer(math.log(40.0)))
        self.log_k1     = self.add_weight(name='log_k1',
                            shape=(), initializer=tf.constant_initializer(math.log(8000.0)))
        self.log_c1     = self.add_weight(name='log_c1',
                            shape=(), initializer=tf.constant_initializer(math.log(7.0)))
        self.log_k2     = self.add_weight(name='log_k2',
                            shape=(), initializer=tf.constant_initializer(math.log(400000.0)))
        super().build(input_shape)

    def dynamics(self, x, u):
        # Khôi phục giá trị thực dương bằng hàm e^x
        R      = tf.exp(self.log_R)
        kb     = tf.exp(self.log_kb)
        kt     = tf.exp(self.log_kt)

        # Tính toán tổng nội bộ từ hai thành phần độc lập đã bóc tách
        cm     = tf.exp(self.log_cm)
        cb     = tf.exp(self.log_cb)
        C_sum  = cm + cb

        tau_fm = tf.exp(self.log_tau_fm)
        cq     = tf.exp(self.log_cq)
        k1     = tf.exp(self.log_k1)
        c1     = tf.exp(self.log_c1)
        k2     = tf.exp(self.log_k2)

        # Trích xuất vector trạng thái
        i_s        = x[:, 0:1]
        theta_m_s  = x[:, 1:2]
        q_s        = x[:, 2:3]
        dtheta_m_s = x[:, 3:4]
        dq_s       = x[:, 4:5]

        Vs      = u[:, 0:1]
        tau_ext = u[:, 1:2]

        # Sai lệch đàn hồi góc xoắn trục
        delta_theta = theta_m_s / self.N - q_s

        # Làm mịn hàm sign bằng hàm tanh có độ dốc vừa phải (50.0) nhằm bảo toàn Gradient cho Adam
        sign_dtheta = tf.math.tanh(50.0 * dtheta_m_s)

        # Hệ phương trình trạng thái động lực học lý thuyết vi phân
        di_dt = ((-R / self.L) * i_s - (kb / self.L) * dtheta_m_s + (1.0 / self.L) * Vs)

        dtheta_m_dt = dtheta_m_s
        dq_dt       = dq_s

        # ĐỒNG NHẤT DẤU TRỪ (-) CHO THÀNH PHẦN PHI TUYẾN K2 KHỚP VỚI MATLAB
        ddtheta_m_dt = ( (kt / self.J_sum) * i_s
                        - (k1 * self.iHD**2 / self.J_sum) * theta_m_s
                        + (k1 * self.iHD    / self.J_sum) * q_s
                        - ((C_sum + c1 * self.iHD**2) / self.J_sum) * dtheta_m_s
                        + (c1 * self.iHD / self.J_sum) * dq_s
                        - (k2 * self.iHD**2 / self.J_sum) * delta_theta**3
                        - (tau_fm / self.J_sum) * sign_dtheta )

        ddq_dt = ( (k1 * self.iHD / self.Jq) * theta_m_s
                  - (k1 / self.Jq) * q_s
                  + (c1 * self.iHD / self.Jq) * dtheta_m_s
                  - ((cq + c1) / self.Jq) * dq_s
                  - (k2 * self.iHD / self.Jq) * delta_theta**3
                  - (1.0 / self.Jq) * tau_ext )

        return tf.concat([di_dt, dtheta_m_dt, dq_dt, ddtheta_m_dt, ddq_dt], axis=-1)

    def call(self, inputs, states):
        x_prev = states[0]
        u_curr = inputs

        # Thuật toán tích phân số Runge-Kutta bậc 4 (RK4)
        k1 = self.dynamics(x_prev, u_curr)
        k2 = self.dynamics(x_prev + 0.5 * self.dt * k1, u_curr)
        k3 = self.dynamics(x_prev + 0.5 * self.dt * k2, u_curr)
        k4 = self.dynamics(x_prev + self.dt * k3, u_curr)

        x_next = x_prev + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next, [x_next]

# =====================================================================
# 4. HÀM LOSS WMSE CÂN BẰNG TRỌNG SỐ CHO CÁC BIẾN TRẠNG THÁI
# =====================================================================
x_max = np.max(np.abs(x_all), axis=0)
y_max = np.max(x_max)
W = (y_max / (x_max + 1e-8)).astype(np.float32)

print(f"\nTrọng số WMSE: {W}")
loss_weights = tf.constant(W, dtype=tf.float32)

def weighted_mse_loss(y_true, y_pred):
    err = (y_true - y_pred) * loss_weights
    return tf.reduce_mean(tf.square(err))

# =====================================================================
# 5. XÂY DỰNG VÀ KHỞI CHẠY QUÁ TRÌNH TRAINING MẠNG PINN
# =====================================================================
model3_cell     = Model3RK4Cell(dt=dt)
pinn_rnn_layer  = RNN(model3_cell, return_sequences=True)

input_u  = tf.keras.Input(shape=(window_size - 1, 2), name="input_u")
input_x0 = tf.keras.Input(shape=(5,),                 name="input_x0")

outputs = pinn_rnn_layer(input_u, initial_state=[input_x0])
model   = tf.keras.Model(inputs=[input_u, input_x0], outputs=outputs)

# Bộ Callback giúp giảm Learning Rate tự động nếu hàm Loss đi vào vùng bão hòa
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1
    )
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_mse_loss
)

print("\n=== BẮT ĐẦU HUÂN LUYỆN H-PINN MODEL 3 (300 EPOCHS) ===")
history = model.fit(
    {"input_u": u_train, "input_x0": x_inits},
    x_train,
    epochs=300,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# =====================================================================
# 6. IN KẾT QUẢ SỐ LIỆU NHẬN DẠNG BÓC TÁCH cm VÀ cb KHÁCH QUAN
# =====================================================================
print("\n=========================================")
print("   KẾT QUẢ NHẬN DẠNG THAM SỐ MODEL 3")
print("=========================================")
print(f"R      (Truth: 0.4932)   -> PINN: {tf.exp(model3_cell.log_R).numpy():.4f}")
print(f"kb     (Truth: 0.05794)  -> PINN: {tf.exp(model3_cell.log_kb).numpy():.5f}")
print(f"kt     (Truth: 0.05507)  -> PINN: {tf.exp(model3_cell.log_kt).numpy():.5f}")
print(f"cm     (Truth: 0.03040)  -> PINN: {tf.exp(model3_cell.log_cm).numpy():.5f}")  # Đã hiển thị riêng cm
print(f"cb     (Truth: 0.0001459)-> PINN: {tf.exp(model3_cell.log_cb).numpy():.6f}")  # Đã hiển thị riêng cb
print(f"tau_fm (Truth: 0.04464)  -> PINN: {tf.exp(model3_cell.log_tau_fm).numpy():.5f}")
print(f"cq     (Truth: 40.31)    -> PINN: {tf.exp(model3_cell.log_cq).numpy():.4f}")
print(f"k1     (Truth: 8000)     -> PINN: {tf.exp(model3_cell.log_k1).numpy():.2f}")
print(f"c1     (Truth: 7.1)      -> PINN: {tf.exp(model3_cell.log_c1).numpy():.4f}")
print(f"k2     (Truth: 400000)   -> PINN: {tf.exp(model3_cell.log_k2).numpy():.2f}")
print("=========================================\n")

# =====================================================================
# 7. HÀM GIẢ LẬP ĐỂ KIỂM TRA ĐỒ THỊ (CẤU HÌNH 10 PHẦN TỬ CHO cm VÀ cb)
# =====================================================================
def forward_sim_model3(params, u_data, dt_s):
    R_v, kb_v, kt_v, cm_v, cb_v, tau_fm_v, cq_v, k1_v, c1_v, k2_v = params
    L_c     = 0.025
    J_sum_c = 7.135e-4
    Jq_c    = 2.7780
    N_c     = 101.0
    iHD_c   = 1.0 / N_c
    C_sum_v = cm_v + cb_v  # Tổng ma sát tổng hợp nội bộ trong hàm giải số

    n = u_data.shape[0]
    x = np.zeros((n, 5))
    x[0, :] = x_all[0].copy()

    for k in range(n - 1):
        Vs_k  = u_data[k, 0]
        tau_k = u_data[k, 1]

        def dyn(s):
            i_s, th_s, q_s, dth_s, dq_s = s
            delta = th_s / N_c - q_s
            smooth_sign = np.tanh(50.0 * dth_s)

            di   = (-R_v/L_c)*i_s - (kb_v/L_c)*dth_s + (1/L_c)*Vs_k
            dth  = dth_s
            dq   = dq_s

            ddth = ((kt_v/J_sum_c)*i_s
                    - (k1_v*iHD_c**2/J_sum_c)*th_s
                    + (k1_v*iHD_c/J_sum_c)*q_s
                    - ((C_sum_v + c1_v*iHD_c**2)/J_sum_c)*dth_s
                    + (c1_v*iHD_c/J_sum_c)*dq_s
                    - (k2_v*iHD_c**2/J_sum_c)*delta**3
                    - (tau_fm_v/J_sum_c)*smooth_sign)

            ddq  = ((k1_v*iHD_c/Jq_c)*th_s
                    - (k1_v/Jq_c)*q_s
                    + (c1_v*iHD_c/Jq_c)*dth_s
                    - ((cq_v+c1_v)/Jq_c)*dq_s
                    - (k2_v*iHD_c/Jq_c)*delta**3
                    - tau_k/Jq_c)
            return np.array([di, dth, dq, ddth, ddq])

        xk = x[k].copy()
        f1 = dyn(xk)
        f2 = dyn(xk + 0.5*dt_s*f1)
        f3 = dyn(xk + 0.5*dt_s*f2)
        f4 = dyn(xk + dt_s*f3)
        x[k+1] = xk + (dt_s/6.0)*(f1 + 2*f2 + 2*f3 + f4)

    return x

# Chạy mô phỏng kiểm chứng với cấu trúc mảng mới chứa 10 biến độc lập
params_init = [1.15, 0.095, 0.045, 3.0e-2, 1.4e-4, 0.05, 40.0, 8000.0, 7.0, 400000.0]
x_initial   = forward_sim_model3(params_init, u_all, dt)

params_pinn = [tf.exp(model3_cell.log_R).numpy(),
               tf.exp(model3_cell.log_kb).numpy(),
               tf.exp(model3_cell.log_kt).numpy(),
               tf.exp(model3_cell.log_cm).numpy(),  # cm
               tf.exp(model3_cell.log_cb).numpy(),  # cb
               tf.exp(model3_cell.log_tau_fm).numpy(),
               tf.exp(model3_cell.log_cq).numpy(),
               tf.exp(model3_cell.log_k1).numpy(),
               tf.exp(model3_cell.log_c1).numpy(),
               tf.exp(model3_cell.log_k2).numpy()]
x_pinn      = forward_sim_model3(params_pinn, u_all, dt)
x_truth     = x_all

# Tính toán giá trị Delta_Theta
N_const = 101.0
delta_theta_truth   = x_truth[:, 1] / N_const - x_truth[:, 2]
delta_theta_initial = x_initial[:, 1] / N_const - x_initial[:, 2]
delta_theta_pinn    = x_pinn[:, 1] / N_const - x_pinn[:, 2]

x_truth_plot   = np.hstack((x_truth, delta_theta_truth.reshape(-1, 1)))
x_initial_plot = np.hstack((x_initial, delta_theta_initial.reshape(-1, 1)))
x_pinn_plot    = np.hstack((x_pinn, delta_theta_pinn.reshape(-1, 1)))

# =====================================================================
# 8. ĐÁNH GIÁ CHI TIẾT SAI SỐ ĐẦU RA (MAE & RMSE)
# =====================================================================
state_names_extended  = ['Current (A)', 'Theta_m (rad)', 'Q_load (rad)', 'Speed (rad/s)', 'DotQ (rad/s)', 'Delta_Theta (rad)']

print("--- Sai số PINN mô phỏng lại so với Dữ liệu CSV gốc ---")
for j in range(x_truth.shape[1]):
    name = state_names_extended[j]
    mae  = np.mean(np.abs(x_truth[:, j] - x_pinn[:, j]))
    rmse = np.sqrt(np.mean(np.square(x_truth[:, j] - x_pinn[:, j])))
    print(f"{name:20s} -> MAE: {mae:.6f} | RMSE: {rmse:.6f}")

mae_delta_theta   = np.mean(np.abs(delta_theta_truth - delta_theta_pinn))
rmse_delta_theta  = np.sqrt(np.mean(np.square(delta_theta_truth - delta_theta_pinn)))
print(f"{state_names_extended[5]:20s} -> MAE: {mae_delta_theta:.6f} | RMSE: {rmse_delta_theta:.6f}")

# =====================================================================
# 9. VẼ ĐỒ THỊ SO SÁNH TRỰC QUAN ĐỒNG BỘ ĐỊNH DẠNG TIMES NEW ROMAN
# =====================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "Liberation Serif"]
plt.rcParams["font.size"]   = 12

plot_idx      = [3, 4, 5]
plot_labels   = [r'$\dot{\theta}_m$ (rad/s)', r'$\dot{q}$ (rad/s)', r'$\Delta\theta$ (rad)']
legend_locs_states = ['lower right', 'lower right', 'upper left']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# Khung đồ thị 1: Đường cong Loss hội tụ
axes[0].plot(history.history['loss'], color='blue', linewidth=2, label='WMSE Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss Value', fontsize=12)
axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[0].grid(True, linestyle='--')
axes[0].legend(loc='upper right', prop={'size': 11})

# Khung đồ thị 2, 3, 4: So sánh 3 biến trạng thái cơ học quan trọng nhất
for i, (idx, ylabel, loc) in enumerate(zip(plot_idx, plot_labels, legend_locs_states)):
    ax = axes[i + 1]
    ax.plot(t, x_truth_plot[:, idx],   'b-',  label='Truth (From CSV)', linewidth=2)
    ax.plot(t, x_initial_plot[:, idx], 'r--', label='Initial Guess',     linewidth=1.5)
    ax.plot(t, x_pinn_plot[:, idx],    'k:',  label='H-PINN (Optimized)', linewidth=2, marker='^', markevery=40)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.legend(loc=loc, prop={'size': 11})
    ax.grid(True, linestyle='--')

plt.tight_layout()
plt.show()
