function model3_final()
    clear; clc; close all;

    % =====================================================================
    % 1. THAM SỐ TABLE III — TRUTH
    % =====================================================================
    R       = 0.4932;
    L       = 0.025;
    kb      = 0.05794;
    kt      = 0.05507;

    Jr      = 0.00065;
    Jm      = 6.5e-5;
    J_sum   = Jr + Jm;

    cb      = 1.459e-4;
    cm      = 3.040e-2;
    C_sum   = cb + cm;
    tau_fm  = 0.04464;

    Jq      = 2.7780;
    cq      = 40.31;
    tau_fq  = 0.0;

    k1      = 8000;
    c1      = 7.1;
    k2      = 400000;
    N       = 101;
    iHD     = 1/N;

    Vs      = 12.0;
    tau_L   = 0.0;

    p = [R; L; kb; kt; J_sum; C_sum; tau_fm; Jq; cq; tau_fq; ...
         k1; c1; k2; N; iHD; Vs; tau_L];

    % =====================================================================
    % 2. ODE45
    % State: [i, theta_m, q, dot_theta_m, dot_q]
    % =====================================================================
    tspan   = 0:0.001:1;
    y0      = zeros(5,1);
    options = odeset('RelTol',1e-6,'AbsTol',1e-8);

    [t, states] = ode45(@(t,y) odefun(t, y, p), tspan, y0, options);

    % =====================================================================
    % 3. TRÍCH XUẤT
    % =====================================================================
    i_sim       = states(:,1);
    theta_m     = states(:,2);
    q_load      = states(:,3);
    dot_theta_m = states(:,4);
    dot_q       = states(:,5);

    tau_m_dynamic = kt * i_sim;
    delta_theta   = theta_m/N - q_load;   % eq.7: Δθ = θ_m/N - q

    % =====================================================================
    % 4. VẼ ĐỒ THỊ
    % =====================================================================
    set(0,'DefaultAxesFontName',   'Times New Roman');
    set(0,'DefaultAxesFontSize',   12);
    set(0,'DefaultTextFontName',   'Times New Roman');
    set(0,'DefaultTextFontSize',   12);
    set(0,'DefaultLegendFontName', 'Times New Roman');

    figure('Name','Model 3 — Final', 'Position',[100 100 950 650]);

    subplot(2,2,1);
    plot(t, Vs*ones(size(t)), 'b', 'LineWidth', 2); hold on;
    plot(t, tau_m_dynamic,    'r', 'LineWidth', 2);
    title('System Inputs','FontSize',13,'FontWeight','bold');
    xlabel('Time (s)','FontSize',12);
    legend('$V_s$ (V)','$\tau_m$ (Nm)', ...
           'Interpreter','latex','Location','best','FontSize',11);
    ylim([0 14]); grid on;

    subplot(2,2,2);
    plot(t, dot_theta_m, 'b', 'LineWidth', 2); hold on;
    plot(t, dot_q,       'r', 'LineWidth', 2);
    title('System Velocity','FontSize',13,'FontWeight','bold');
    xlabel('Time (s)','FontSize',12);
    legend('$\dot{\theta}_m$ (rad/s)','$\dot{q}$ (rad/s)', ...
           'Interpreter','latex','Location','best','FontSize',11);
    ylim([0 35]); grid on;

    subplot(2,2,3);
    plot(t, i_sim, 'b', 'LineWidth', 2);
    title('Current','FontSize',13,'FontWeight','bold');
    xlabel('Time (s)','FontSize',12);
    ylabel('$i$ (A)','Interpreter','latex','FontSize',12);
    ylim([0 25]); grid on;

    subplot(2,2,4);
    plot(t, delta_theta, 'b', 'LineWidth', 2);
    title('Torsional Difference','FontSize',13,'FontWeight','bold');
    xlabel('Time (sec)','FontSize',12);
    ylabel('$\Delta\theta$ (rad)','Interpreter','latex','FontSize',12);
    grid on;

    exportgraphics(gcf,'model3_fig4.png','Resolution',300);

    % =====================================================================
    % 5. XUẤT CSV
    % =====================================================================
    T = table(t, i_sim, theta_m, q_load, dot_theta_m, dot_q, ...
              delta_theta, tau_m_dynamic, ...
              'VariableNames', {'Time','Current','Theta_m','Q_load', ...
                                'Speed','DotQ','Delta_Theta','Tau_m'});
    writetable(T,'model3_simulation.csv');
    fprintf('Xuất CSV xong: %d dòng x %d cột\n', height(T), width(T));

    % In giá trị xác lập
    fprintf('\n=== Xác lập (t = 1s) ===\n');
    fprintf('i          = %.4f A\n',       i_sim(end));
    fprintf('dot_theta_m= %.4f rad/s\n',   dot_theta_m(end));
    fprintf('dot_q      = %.4f rad/s\n',   dot_q(end));
    fprintf('tau_m      = %.4f Nm\n',      tau_m_dynamic(end));
    fprintf('delta_theta= %.4e rad\n',     delta_theta(end));
end

% =====================================================================
% HÀM ĐỘNG LỰC HỌC
% Eq.(12): q3_dot = A3*q3 + B3*u3 + C3*N(Δθ³, sign(θ̇_m), sign(q̇))
% =====================================================================
function dq = odefun(~, q, p)
    R=p(1);  L=p(2);   kb=p(3);    kt=p(4);
    J_sum=p(5);  C_sum=p(6);  tau_fm=p(7);
    Jq=p(8); cq=p(9);  tau_fq=p(10);
    k1=p(11); c1=p(12); k2=p(13);
    N=p(14);  iHD=p(15); Vs=p(16); tau_L=p(17);

    % --- A3: ma trận tuyến tính 5x5 ---
    % iHD = 1/N → iHD/N = 1/N²
    A3 = [ -R/L,              0,           0,       -kb/L,              0;
            0,                0,           0,          1,               0;
            0,                0,           0,          0,               1;
            kt/J_sum,  -k1*iHD^2/J_sum,  k1*iHD/J_sum, -(C_sum+c1*iHD^2)/J_sum,  c1*iHD/J_sum;
            0,          k1*iHD/Jq,      -k1/Jq,    c1*iHD/Jq,       -(cq+c1)/Jq ];

    % --- B3: ma trận đầu vào ---
    B3 = [ 1/L,   0;
            0,    0;
            0,    0;
            0,    0;
            0,  -1/Jq ];

    % --- C3: ma trận phi tuyến ---
    % Hàng 4: k2*iHD²/J_sum nhất quán với A3
    % Hàng 5: k2*iHD/Jq
    C3 = [      0,              0,           0;
                0,              0,           0;
                0,              0,           0;
           -k2*iHD^2/J_sum,  -tau_fm/J_sum,  0;
           -k2*iHD/Jq,            0,     -tau_fq/Jq ];

    % Δθ = θ_m/N - q  (eq.7)
    delta_theta = q(2)/N - q(3);

    N_yt = [ delta_theta^3;
             sign(q(4));
             sign(q(5)) ];

    u3 = [Vs; tau_L];
    dq = A3*q + B3*u3 + C3*N_yt;
end