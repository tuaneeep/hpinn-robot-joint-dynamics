import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
import matplotlib.pyplot as plt

df = pd.read_csv('bldc_dataset.csv')
# Đọc tập dữ liệu
print(df.head())

# Trích xuất dữ liệu sang mảng NumPy dạng float32
t = df['Time'].values.astype(np.float32)
i_data = df['Current'].values.astype(np.float32)
omega_data = df['Speed'].values.astype(np.float32)

# Tính toán bước thời gian dt trung bình
dt = np.mean(np.diff(t))
print(f"\nBước thời gian dt của dữ liệu: {dt:.6f} giây")

# Tạo đầu vào u = [Vs, tau_m] (Vs = 12V, tau_m = 0)
Vs_data = 12.0 * np.ones_like(t)
tau_m_data = 0.0 * np.ones_like(t)

u_all = np.stack([Vs_data, tau_m_data], axis=-1)
x_all = np.stack([i_data, omega_data], axis=-1)


window_size = 40
step_size = 5

u_train_list = []
x_train_list = []
x_init_list = []  # Lưu điều kiện biên thực tế tại điểm bắt đầu của từng cửa sổ

for i in range(0, len(t) - window_size, step_size):
    u_block = u_all[i : i + window_size - 1, :]
    x_block = x_all[i + 1 : i + window_size, :]

    u_train_list.append(u_block)
    x_train_list.append(x_block)
    x_init_list.append(x_all[i])  # Điểm xuất phát thực tế của phân đoạn thứ i

# Chuyển đổi sang Tensor định dạng chuẩn mạng RNN: [Batch_Size, Time_Steps, Features]
u_train = tf.convert_to_tensor(u_train_list, dtype=tf.float32)
x_train = tf.convert_to_tensor(x_train_list, dtype=tf.float32)
x_inits = tf.convert_to_tensor(x_init_list, dtype=tf.float32)

print(f"Kích thước Tensor Đầu vào mới (u_train): {u_train.shape}")
print(f"Kích thước Tensor Đầu ra mới  (x_train): {x_train.shape}")
print(f"Kích thước Tensor Điều kiện biên (x_inits): {x_inits.shape}")


# --- CẤU TRÚC RK4 CELL CHUẨN VẬT LÝ MẠNG RNN ---
class BLDCRK4Cell(Layer):
    def __init__(self, dt, **kwargs):
        super(BLDCRK4Cell, self).__init__(**kwargs)
        self.dt = dt
        self.state_size = 2 # Trạng thái [i, omega]

    def build(self, input_shape):
        self.R = self.add_weight(name='R', shape=(), initializer=tf.keras.initializers.Constant(1.15), constraint=tf.keras.constraints.NonNeg())
        self.cb = self.add_weight(name='cb', shape=(), initializer=tf.keras.initializers.Constant(0.0003), constraint=tf.keras.constraints.NonNeg())
        self.kb = self.add_weight(name='kb', shape=(), initializer=tf.keras.initializers.Constant(0.095), constraint=tf.keras.constraints.NonNeg())
        self.kt = self.add_weight(name='kt', shape=(), initializer=tf.keras.initializers.Constant(0.045), constraint=tf.keras.constraints.NonNeg())
        self.tau_fc = self.add_weight(name='tau_fc', shape=(), initializer=tf.keras.initializers.Constant(0.05), constraint=tf.keras.constraints.NonNeg())

        self.L = tf.constant(0.025, dtype=tf.float32)
        self.Jr = tf.constant(0.00065, dtype=tf.float32)

        super(BLDCRK4Cell, self).build(input_shape)

    def bldc_dynamics_tf(self, x, u):
        i = x[:, 0:1]
        omega = x[:, 1:2]
        Vs = u[:, 0:1]
        tau_m = u[:, 1:2]

        # Hệ phương trình vi phân động cơ BLDC
        di_dt = (-self.R / self.L) * i - (self.kb / self.L) * omega + (1.0 / self.L) * Vs


        coulomb_sign = tf.math.tanh(500.0 * omega)
        domega_dt = (self.kt / self.Jr) * i - (self.cb / self.Jr) * omega - (tau_m / self.Jr) - (self.tau_fc / self.Jr) * coulomb_sign

        return tf.concat([di_dt, domega_dt], axis=-1)

    def call(self, inputs, states):
        x_prev = states[0]
        u_curr = inputs

        k1 = self.bldc_dynamics_tf(x_prev, u_curr)
        k2 = self.bldc_dynamics_tf(x_prev + 0.5 * self.dt * k1, u_curr)
        k3 = self.bldc_dynamics_tf(x_prev + 0.5 * self.dt * k2, u_curr)
        k4 = self.bldc_dynamics_tf(x_prev + self.dt * k3, u_curr)

        x_next = x_prev + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next, [x_next]


# --- THIẾT LẬP HÀM LOSS WMSE ---
i_max = np.max(np.abs(i_data))
omega_max = np.max(np.abs(omega_data))
y_max = max(i_max, omega_max)

W_current = y_max / i_max
W_speed = y_max / omega_max

print(f"\n--- Trọng số WMSE: W_i = ymax / yi_max ---")
print(f"W_current: {W_current:.4f}")
print(f"W_speed:   {W_speed:.4f}")

loss_weights = tf.constant([W_current, W_speed], dtype=tf.float32)

def weighted_mse_loss(y_true, y_pred):
    error_weighted = (y_true - y_pred) * loss_weights
    return tf.reduce_mean(tf.square(error_weighted))



bldc_cell = BLDCRK4Cell(dt=dt)
pinn_rnn_layer = RNN(bldc_cell, return_sequences=True)

# Khai báo đa đầu vào (Input_u: chuỗi tín hiệu điều khiển, Input_x0: trạng thái xuất phát của từng Window)
input_u = tf.keras.Input(shape=(window_size - 1, 2), name="input_u")
input_x0 = tf.keras.Input(shape=(2,), name="input_x0")


outputs = pinn_rnn_layer(input_u, initial_state=[input_x0])

model = tf.keras.Model(inputs=[input_u, input_x0], outputs=outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_mse_loss
)

print("\nBẮT ĐẦU HUẤN LUYỆN")

history = model.fit(
    {"input_u": u_train, "input_x0": x_inits},
    x_train,
    epochs=350,
    batch_size=16,
    verbose=1
)


print("\n=========================================")
print("KẾT QUẢ NHẬN DẠNG THAM SỐ\n")
print(f"Tham số R      (Truth: 0.4932)   -> PINN : {bldc_cell.R.numpy():.4f}")
print(f"Tham số cb     (Truth: 0.0002537)-> PINN : {bldc_cell.cb.numpy():.7f}")
print(f"Tham số kb     (Truth: 0.05794)  -> PINN : {bldc_cell.kb.numpy():.5f}")
print(f"Tham số kt     (Truth: 0.05507)  -> PINN : {bldc_cell.kt.numpy():.5f}")
print(f"Tham số tau_fc (Truth: 0.04464)  -> PINN : {bldc_cell.tau_fc.numpy():.5f}")
print("=========================================\n")



def forward_simulation(R_val, cb_val, kb_val, kt_val, tau_fc_val, u_data, dt_sampling):
    num_steps = u_data.shape[0]
    x_sim = np.zeros((num_steps, 2))

    # Gán trực tiếp điểm xuất phát thời gian t=0 bằng dữ liệu thực tế
    x_sim[0, :] = x_all[0].copy()

    L_const = 0.025
    Jr_const = 0.00065

    for k in range(num_steps - 1):
        Vs = u_data[k, 0]
        tau_m = u_data[k, 1]

        def dynamics(x_s):
            i_s, omega_s = x_s[0], x_s[1]
            di = (-R_val / L_const) * i_s - (kb_val / L_const) * omega_s + (1.0 / L_const) * Vs


            dw = (kt_val / Jr_const) * i_s - (cb_val / Jr_const) * omega_s - (tau_m / Jr_const) - (tau_fc_val / Jr_const) * np.sign(omega_s)
            return np.array([di, dw])

        # Lấy dữ liệu mốc hiện tại để tích phân lên mốc kế tiếp k+1
        x_curr = x_sim[k, :].copy()
        k1 = dynamics(x_curr)
        k2 = dynamics(x_curr + 0.5 * dt_sampling * k1)
        k3 = dynamics(x_curr + 0.5 * dt_sampling * k2)
        k4 = dynamics(x_curr + dt_sampling * k3)

        x_sim[k+1, :] = x_curr + (dt_sampling / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_sim


x_truth = x_all

# Tính toán các quỹ đạo phản hồi động học
x_initial = forward_simulation(1.15, 0.0003, 0.095, 0.045, 0.05, u_all, dt)

R_p = bldc_cell.R.numpy()
cb_p = bldc_cell.cb.numpy()
kb_p = bldc_cell.kb.numpy()
kt_p = bldc_cell.kt.numpy()
tau_p = bldc_cell.tau_fc.numpy()
x_pinn = forward_simulation(R_p, cb_p, kb_p, kt_p, tau_p, u_all, dt)


# --- ĐÁNH GIÁ CHỈ SỐ SAI SỐ ---
mae_current = np.mean(np.abs(x_truth[:, 0] - x_pinn[:, 0]))
rmse_current = np.sqrt(np.mean(np.square(x_truth[:, 0] - x_pinn[:, 0])))
mae_speed = np.mean(np.abs(x_truth[:, 1] - x_pinn[:, 1]))
rmse_speed = np.sqrt(np.mean(np.square(x_truth[:, 1] - x_pinn[:, 1])))


print(f"Dòng điện (Current) -> MAE: {mae_current:.4f} A  | RMSE: {rmse_current:.4f} A")
print(f"Tốc độ (Speed)     -> MAE: {mae_speed:.4f} rad/s | RMSE: {rmse_speed:.4f} rad/s")


# --- ĐOẠN MÃ VẼ ĐỒ THỊ ĐÃ ĐƯỢC CẤU HÌNH LẠI ---

# Cấu hình font hệ thống sang Times New Roman cho toàn bộ biểu đồ
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

fig = plt.figure(figsize=(15, 6))

# Ô 1 (Cột trái): Đường cong hội tụ Loss (Cắt hẹp, chiếm 1 cột trên lưới 3 cột)
ax_loss = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
ax_loss.plot(history.history['loss'], color='blue', linewidth=2, label='WMSE Loss')
ax_loss.set_xlabel('Epoch', fontsize=12)
ax_loss.set_ylabel('Loss Value', fontsize=12)
ax_loss.grid(True, linestyle='--')
ax_loss.legend(prop={'size': 11})

# Ô 2 (Cột phải - Dòng trên): Biểu đồ Dòng điện (Current) (Mở rộng diện tích)
ax1 = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
ax1.plot(t, x_truth[:, 0], 'b-', label='Truth', linewidth=2)
ax1.plot(t, x_initial[:, 0], 'r--', label='Initial Guess', linewidth=1.5)
ax1.plot(t, x_pinn[:, 0], 'k:', label='H-PINN', linewidth=2, marker='^', markevery=40)
ax1.set_ylabel('Current (A)', fontsize=12)
ax1.legend(loc='upper right', prop={'size': 11})
ax1.grid(True, linestyle='--')

# Ô 3 (Cột phải - Dòng dưới): Biểu đồ Tốc độ (Speed) (Mở rộng diện tích)
ax2 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
ax2.plot(t, x_truth[:, 1], 'b-', label='Truth', linewidth=2)
ax2.plot(t, x_initial[:, 1], 'r--', label='Initial Guess', linewidth=1.5)
ax2.plot(t, x_pinn[:, 1], 'k:', label='H-PINN', linewidth=2, marker='^', markevery=40)
ax2.set_ylabel('Speed (rad/s)', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.legend(loc='lower right', prop={'size': 11})
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()
