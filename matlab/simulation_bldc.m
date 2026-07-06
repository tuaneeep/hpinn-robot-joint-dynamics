function generate_bldc_data()

    clear; clc; close all;

    params.R      = 4.932e-1;
    params.cb     = 2.537e-4;
    params.kb     = 5.794e-2;
    params.kt     = 5.507e-2;
    params.tau_fc = 4.464e-2;
    params.L      = 0.025;
    params.Jr     = 0.00065;

% input
    params.Vs    = 12.0;     
    params.tau_m = 0.0;     

   
    dt_sampling = 0.002;
    t_span = 0 : dt_sampling : 1.0;

    q0 = [0;0];

    options = odeset( ...
        'RelTol',1e-6,...
        'AbsTol',1e-8);


    [t,q] = ode45( ...
        @(t,q)bldc_dynamics(t,q,params), ...
        t_span, ...
        q0, ...
        options);

    i     = q(:,1);
    omega = q(:,2);

   

    Voltage    = params.Vs    * ones(size(t));
    LoadTorque = params.tau_m * ones(size(t));

    dataset = table( ...
        t,...
        i,...
        omega,...
        'VariableNames',...
        {'Time','Current','Speed'});

    % LƯU FILE
   
    writetable(dataset,'bldc_dataset.csv');

    save('bldc_step_data.mat',...
         't','i','omega');

    fprintf('CSV : bldc_dataset.csv\n');
    fprintf('MAT : bldc_step_data.mat\n');
    fprintf('So mau: %d\n',height(dataset));

    % VE DO THI

    figure;

    subplot(2,1,1)
    plot(t,i,'b','LineWidth',1.5)
    grid on
    xlabel('Time (s)')
    ylabel('Current (A)')
    title('Current Response')

    subplot(2,1,2)
    plot(t,omega,'r','LineWidth',1.5)
    grid on
    xlabel('Time (s)')
    ylabel('Speed (rad/s)')
    title('Speed Response')

end


function dq = bldc_dynamics(~, q, params)

    i     = q(1);
    omega = q(2);

    di_dt = ...
        (-params.R/params.L)*i ...
        - (params.kb/params.L)*omega ...
        + (1/params.L)*params.Vs;

    domega_dt = ...
        (params.kt/params.Jr)*i ...
        - (params.cb/params.Jr)*omega ...
        - (params.tau_m/params.Jr) ...
        - (params.tau_fc/params.Jr)*sign(omega);

    dq = [di_dt ; domega_dt];

end