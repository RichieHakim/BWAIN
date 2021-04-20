length_trace = 10000;
length_kernel = 100;
t_kernel = [1:length_kernel];
tau = 8;

t_alpha = t_kernel.*exp(-t_kernel/tau);

figure; plot(t_alpha)

x = (rand(length_trace,1) > 0.99) .* abs(randn(length_trace,1).^1.5);

figure; plot(x)

% x_conv = conv(x,t_alpha);
x_conv = conv(x,t_alpha);
x_conv_noise = x_conv + randn(numel(x_conv),1)/2;

figure; plot(x_conv_noise)

%%
length_kernel = 100;
t_kernel = [1:length_kernel];
tau = 8;

t_alpha = t_kernel.*exp(-t_kernel/tau);

length_history = 100;
t_history = 1:length_history;

clear output
for jj = 1:4
    cc = 1;
    x = zeros(length_history,1);
%     for ii = 1:108000
    for ii = 1:27000
        if numel(x) < length_history
            x = zeros(length_history,1);
        end
        if numel(x) > length_history
            x(1:end-length_history) = [];
        end
        
        hyst_vector = fliplr(100000.^(1-t_history./length_history) / 100000.^(1));
        x(end+1) = (rand(1) > (1-(0.007 + (sum(x.*hyst_vector') * .05)))) *abs(randn(1)^2);
        %     figure; plot(x)
        
        x_conv = conv(x, t_alpha);
        
        %     output(cc) = x_conv(100);
        output{jj}(cc) = (x_conv(length_history) + randn(1)*1) / 15;
        cc = cc+1;
    end
    figure; plot(output{jj})
end

