interaction = load('../data_ddt/drdi_matrix.txt');
nFold = 5;
rng(1);
Pint = find(interaction); % pair of interaction
Nint = length(Pint);
Pnoint = find(~interaction);
Pnoint = Pnoint(randperm(length(Pnoint), Nint * 1));
Nnoint = length(Pnoint);

posFilt = crossvalind('Kfold', Nint, nFold);
negFilt = crossvalind('Kfold', Nnoint, nFold);

for foldID = 1 : nFold
    % 生成训练集
    train_posIdx = Pint(posFilt ~= foldID);
    train_negIdx = Pnoint(negFilt ~= foldID);
    train_idx = [train_posIdx; train_negIdx];
    train_matrix = zeros(size(interaction));
    train_matrix(train_idx) = interaction(train_idx);
    train_matrixT = train_matrix';
    fprintf('Train data: %d positives, %d negatives\n', sum(interaction(train_posIdx) == 1), sum(interaction(train_negIdx) == 0));

    % 生成测试集
    test_posIdx = Pint(posFilt == foldID);
    test_negIdx = Pnoint(negFilt == foldID);
    test_idx = [test_posIdx; test_negIdx];
    test_matrix = zeros(size(interaction));
    test_matrix(test_idx) = interaction(test_idx);
    fprintf('Test data: %d positives, %d negatives\n', sum(interaction(test_posIdx) == 1), sum(interaction(test_negIdx) == 0));
    
    % 保存训练集和测试集矩阵
    save(['..\data_ddt\5cv\drdi_train_matrix_fold' num2str(foldID) '.mat'], 'train_matrix');
    save(['..\data_ddt\5cv\didr_train_matrix_fold' num2str(foldID) '.mat'], 'train_matrixT');
     % 保存训练集和测试集矩阵为txt文件
    dlmwrite(['..\data_ddt\5cv\drdi_train_matrix_fold' num2str(foldID) '.txt'], train_matrix, 'delimiter', '\t');
    dlmwrite(['..\data_ddt\5cv\didr_train_matrix_fold' num2str(foldID) '.txt'], train_matrixT, 'delimiter', '\t');
end
