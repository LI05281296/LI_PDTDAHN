% 加载原始交互矩阵
interaction = load('../../data_ddt/drdi_matrix.txt');

% 加载保存的训练矩阵
load('train_matrix_fold1.mat', 'train_matrix');

% 找到正样本的索引
train_posIdx = find(train_matrix);

% 找到负样本的索引
% 负样本是训练矩阵中为零且在原始交互矩阵中也是零的索引
Pnoint = find(~interaction);
train_negIdx = Pnoint(ismember(Pnoint, find(train_matrix == 0)));

% 创建标签向量
Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];

% 合并正负样本的索引
train_idx = [train_posIdx; train_negIdx];

% 输出标签统计信息
fprintf('Train data: %d positives, %d negatives\n', sum(Ytrain == 1), sum(Ytrain == 0));

% 输出原始训练索引
fprintf('Train indices restored, total: %d\n', length(train_idx));
