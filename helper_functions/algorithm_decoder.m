function output = algorithm_decoder(inputs, scale_factors, ensemble_assignments)
E1_inputs = find(ensemble_assignments == 1);
E1_scaling = scale_factors(E1_inputs);

E2_inputs = find(ensemble_assignments == 2);
E2_scaling = scale_factors(E2_inputs);

output = mean(inputs(E1_inputs) .* E1_scaling)  -  mean(inputs(E2_inputs) .* E2_scaling);

% output = mean(inputs);