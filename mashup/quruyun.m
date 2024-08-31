addpath code

%% Required external dependencies: see README.txt for more information
addpath libsvm-package/matlab % LIBSVM package (for cross_validation.m)
%addpath /path-to-lbfgs-package % L-BFGS package (only if svd_approx = false)

%% Example parameters
%org = 'gate';
%folder = 'drug_disease';
%cv = '10_CV_1';
svd_approx = true;  % use SVD approximation for Mashup
                    %   recommended: true for human, false for yeast
ndim = 90;         % number of dimensions
                    %   recommended: 800 for human, 500 for yeast

%% Construct network file paths
string_nets = {'data3'};
network_files = cell(1, length(string_nets));
for i = 1:length(string_nets)
  network_files{i} = sprintf('data/quruyun/%s.txt', ...
                               string_nets{i});         
end

%% Load node list
node_file = sprintf('data/quruyun/node3.txt');
nodes = textread(node_file, '%s');
nnodes = length(nodes);

%% Mashup integration
fprintf('[Mashup]\n');
x = mashup(network_files, nnodes, ndim, svd_approx);
writematrix(x','data/quruyun/result3/result_dimension_90.csv');
