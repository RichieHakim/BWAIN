function output = make_fake_dFoF
persistent x
% clear x

% reward_rate = 0.004;
reward_rate = 0.0010;
% reward_rate = 0.04;
hysteresis_weight = 0.07;
amplitude_nonlinearity = 1.8;
noise_weight = 4;

length_kernel = 100;
t_kernel = [1:length_kernel];
tau = 15;

t_alpha = t_kernel.*exp(-t_kernel/tau);

length_history = 100;
t_history = 1:length_history;
% x
output = zeros(4,1);
for jj = 1:4
    if isempty(x)
        x = {{},{},{},{}};
    end
    if numel(x{jj}) < length_history
        x{jj} = zeros(length_history,1);
    end
    if numel(x{jj}) > length_history
        x{jj}(1:end-length_history) = [];
    end
    
    hyst_vector = fliplr(100000.^(1-t_history./length_history) / 100000.^(1));
    x{jj}(end+1) = (rand(1) > (1-(reward_rate + (sum(x{jj}.*hyst_vector') * hysteresis_weight)))) *abs(randn(1)^amplitude_nonlinearity);
    
    x_conv = conv(x{jj}, t_alpha);
    
    output(jj) = (x_conv(length_history) + randn(1)*noise_weight) / 15;
end
%     figure; plot(output{jj})
end