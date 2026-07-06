function model2_1()
    % Thu dọn bộ nhớ và giao diện
    clc; clear; close all;
    
    % --- TỰ ĐỘNG BẬT INTERPRETER LATEX CHO TOÀN BỘ ĐỒ THỊ ---
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    
    % --- 1. CẤU HÌNH THỜI GIAN VÀ TẦN SỐ LẤY MẪU CHUẨN (500 HZ) ---
    f_s = 500;                  
    dt = 1 / f_s;               
    T_max = 1.5;                
    tspan = 0:dt:T_max; 
    N_samples = length(tspan);  
    
    % Điều kiện ban đầu (Đứng yên)
    q2_init = [0; 0; 0; 0];      
    
    % --- 2. GIẢI PHƯƠNG TRÌNH VI PHÂN BẰNG ODE45 ---
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8); 
    [t, q2] = ode45(@harmonic_dynamics_sub, tspan, q2_init, options);
    
    % --- 3. TRÍCH XUẤT KẾT QUẢ VÀ TÍNH TOÁN ĐẦU VÀO/GIA TỐC GỐC ---
    theta_m_pure   = q2(:, 1);
    q_pure         = q2(:, 2);
    d_theta_m_pure = q2(:, 3); 
    d_q_pure       = q2(:, 4); 
    
    tau_m_vec   = zeros(N_samples, 1);
    tau_ext_vec = zeros(N_samples, 1);
    dd_theta_m_pure  = zeros(N_samples, 1);
    dd_q_pure        = zeros(N_samples, 1);
    
    for i = 1:N_samples
        [dq, tau_m, tau_ext] = harmonic_dynamics_sub(t(i), q2(i, :));
        tau_m_vec(i)   = tau_m;
        tau_ext_vec(i) = tau_ext;
        dd_theta_m_pure(i)  = dq(3); % Gia tốc Motor lý tưởng
        dd_q_pure(i)        = dq(4); % Gia tốc Tải lý tưởng
    end
    
    % =================================================================
    % --- GIẢ LẬP THỰC NGHIỆM: THÊM NHIỄU GAUSSIAN & LỌC BUTTERWORTH ---
    % =================================================================
    
    % 3a. Khởi tạo nhiễu trắng Gaussian (Độ lệch chuẩn vừa phải cho góc và vận tốc)
    rng(42); % Cố định seed để dữ liệu không bị thay đổi ngẫu nhiên mỗi lần chạy
    noise_angle = 0.001 * randn(N_samples, 1); % Nhiễu vị trí
    noise_vel   = 0.015 * randn(N_samples, 1); % Nhiễu vận tốc
    noise_acc   = 0.150 * randn(N_samples, 1); % Nhiễu gia tốc
    
    % Cộng nhiễu vào dữ liệu lý tưởng để tạo "Dữ liệu thô cảm biến"
    theta_m_noisy   = theta_m_pure + noise_angle;
    q_noisy         = q_pure + noise_angle;
    d_theta_m_noisy = d_theta_m_pure + noise_vel;
    d_q_noisy       = d_q_pure + noise_vel;
    dd_theta_m_noisy = dd_theta_m_pure + noise_acc;
    dd_q_noisy       = dd_q_pure + noise_acc;
    
    % 3b. Thiết kế bộ lọc Butterworth bậc 4 kỹ thuật số
    % Tần số cắt (Cutoff frequency) chọn bằng 25 Hz để lọc sạch nhiễu mà không làm trễ pha tín hiệu
    f_cutoff = 25; 
    Wn = f_cutoff / (f_s / 2); % Chuẩn hóa theo tần số Nyquist
    [b, a] = butter(4, Wn, 'low'); % Bộ lọc thông thấp bậc 4
    
    % Lọc làm mịn dữ liệu thô (Dùng 'filtfilt' để triệt tiêu hoàn toàn độ trễ pha)
    theta_m   = filtfilt(b, a, theta_m_noisy);
    q         = filtfilt(b, a, q_noisy);
    d_theta_m = filtfilt(b, a, d_theta_m_noisy);
    d_q       = filtfilt(b, a, d_q_noisy);
    dd_theta_m = filtfilt(b, a, dd_theta_m_noisy);
    dd_q       = filtfilt(b, a, dd_q_noisy);
    
    % =================================================================
    
    % --- 4. XUẤT ĐẦY ĐỦ 9 CỘT DỮ LIỆU ĐÃ QUA XỬ LÝ NHIỄU RA CSV ---
    data_matrix = [t, tau_m_vec, tau_ext_vec, theta_m, q, d_theta_m, d_q, dd_theta_m, dd_q];
    header = {'Time', 'tau_m', 'tau_ext', 'theta_m', 'q', 'd_theta_m', 'd_q', 'dd_theta_m', 'dd_q'};
    
    output_table = array2table(data_matrix, 'VariableNames', header);
    writetable(output_table, 'model2_train_data.csv');
    disp('Đã xuất file dữ liệu thực nghiệm (Có nhiễu & Đã lọc): model2_train_data.csv!');
    
    % --- 5. VẼ ĐỒ THỊ CHUẨN ĐỂ KIỂM TRA ĐỐI CHIẾU ---
    figure('Name', 'Model 2 Simulation with Noise & Filter', 'Position', [100, 50, 1100, 750]);
    
    subplot(2, 2, 1);
    plot(t, d_theta_m_pure, 'g--', 'LineWidth', 1.5); hold on;
    plot(t, d_theta_m, 'b-', 'LineWidth', 2);
    title('$\dot{\theta}_m$', 'FontSize', 12);
    xlabel('Time (s)'); ylabel('$\dot{\theta}_m$ (rad/s)'); grid on; xlim([0, T_max]);
    legend('Pure', 'Filtered');
    
    subplot(2, 2, 2);
    plot(t, d_q_pure, 'g--', 'LineWidth', 1.5); hold on;
    plot(t, d_q, 'b-', 'LineWidth', 2);
    title('$\dot{q}$', 'FontSize', 12);
    xlabel('Time (s)'); ylabel('$\dot{q}$ (rad/s)'); grid on; xlim([0, T_max]);
    legend('Pure', 'Filtered');
    
    subplot(2, 2, 3);
    plot(t, dd_theta_m_pure, 'g--', 'LineWidth', 1.5); hold on;
    plot(t, dd_theta_m, 'r-', 'LineWidth', 1.5);
    title('$\ddot{\theta}_m$', 'FontSize', 12);
    xlabel('Time (s)'); ylabel('$\ddot{\theta}_m$ ($\textrm{rad/s}^2$)'); grid on; xlim([0, T_max]);
    legend('Pure', 'Filtered');
    
    subplot(2, 2, 4);
    plot(t, dd_q_pure, 'g--', 'LineWidth', 1.5); hold on;
    plot(t, dd_q, 'r-', 'LineWidth', 1.5);
    title('$\ddot{q}$', 'FontSize', 12);
    xlabel('Time (s)'); ylabel('$\ddot{q}$ ($\textrm{rad/s}^2$)'); grid on; xlim([0, T_max]);
    legend('Pure', 'Filtered');
end

function [dq2dt, tau_m, tau_ext] = harmonic_dynamics_sub(t, q2)
    theta_m   = q2(1); 
    q         = q2(2); 
    d_theta_m = q2(3); 
    d_q       = q2(4); 
    
    % --- THÔNG SỐ HỆ THỐNG GỐC (GROUND TRUTH) ---
    J_m    = 6e-5;        
    J_q    = 7.780e-1;    
    c_m    = 1.018e-4;    
    c_q    = 1.799;       
    c1     = 18.938;      
    k1     = 5.164e3;     
    k2     = 9.021e8;     
    tau_fm = 2.291e-2;    
    tau_fq = 2.296;       
    
    N = 100;        
    i_HD = 1.0 / N; 
    
    % --- TÁI TẠO QUỸ ĐẠO ĐẦU VÀO TAU_M ---
    tau_ext = 0.0; 
    high_freq_noise = 0.015 * sin(2 * pi * 9 * t) + 0.008 * cos(2 * pi * 23 * t);
    if t < 0.38
        tau_m = 0.145 + high_freq_noise;
    elseif t >= 0.38 && t < 0.88
        tau_m = 0.105 + 0.005 * sin(2 * pi * 5 * t);
    elseif t >= 0.88 && t < 1.22
        tau_m = -0.025 + high_freq_noise;
    else
        tau_m = 0.02 + 0.005 * sin(2 * pi * 4 * t);
    end
    
    % --- CHUẨN HÓA CÔNG THỨC THEO BÀI BÁO PHƯƠNG TRÌNH (7) ---
    delta_theta   = (theta_m / N) - q;
    delta_d_theta = (d_theta_m / N) - d_q;
    
    % Thành phần cốt lõi của lò xo đàn hồi phi tuyến bậc 3
    base_elastic = c1 * delta_d_theta + k1 * delta_theta + k2 * (delta_theta^3);
    
    % --- MA SÁT COULOMB CÓ LỌC CHẶN DẢI CHẾT ---
    if abs(d_theta_m) < 1e-3
        friction_m = tau_fm * (d_theta_m / 1e-3);
    else
        friction_m = tau_fm * sign(d_theta_m);
    end
    
    if abs(d_q) < 1e-3
        friction_q = tau_fq * (d_q / 1e-3);
    else
        friction_q = tau_fq * sign(d_q);
    end
    
    % --- HỆ PHƯƠNG TRÌNH TRẠNG THÁI CHUẨN TUYỆT ĐỐI THEO PT (8) VÀ (9) ---
    dq2dt = zeros(4,1);
    dq2dt(1) = d_theta_m; 
    dq2dt(2) = d_q;       
    dq2dt(3) = (1 / J_m) * (tau_m - c_m * d_theta_m - i_HD * base_elastic - friction_m);
    dq2dt(4) = (1 / J_q) * (-tau_ext - c_q * d_q + base_elastic - friction_q);
end