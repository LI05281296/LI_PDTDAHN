addpath code

%% Required external dependencies: see README.txt for more information
addpath libsvm-package/matlab % LIBSVM package (for cross_validation.m)
%addpath /path-to-lbfgs-package % L-BFGS package (only if svd_approx = false)


%% Example parameters
org = 'multiple_label_data';
% folder = 'drug_disease';
% cv = ['10_CV_',int2str(n)];
svd_approx = true;  % use SVD approximation for Mashup
                    %   recommended: true for human, false for yeast
ndim = 50;         % number of dimensions
                    %   recommended: 800 for human, 500 for yeast
                      
for k = 0:4
    %% Construct network file paths
    string_nets = {['heterogeneous',num2str(k)],'drug_sim','disease_sim','target_sim'};
    network_files = cell(1, length(string_nets));
    for i = 1:length(string_nets)
      network_files{i} = sprintf('data/%s/net_%s.txt', ...
                                  org, string_nets{i});         
    end
    
    %% Load node list
    node_file = sprintf('data/multiple_label_data/string_node.txt');
    nodes = textread(node_file, '%s');
    nnodes = length(nodes);
    
    %% Mashup integration
    fprintf('[Mashup]\n');
    x = mashup(network_files, nnodes, ndim, svd_approx);
    writematrix(x',['multiple_dimension/dimension_50/dimension50_feature_train',num2str(k),'.csv']);
end
