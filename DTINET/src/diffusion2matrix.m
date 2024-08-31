for i =1:5
    % 假设您已经从文件中读取了9个矩阵
    A1 = load('../data_ddt/drug_sim_matrix.txt'); % 124x124
    A2 = load('../data_ddt/disease_sim_matrix.txt'); % 177x177
    A3 = load('../data_ddt/target_sim_matrix.txt'); % 104x104
    A4 = load(['../data_ddt/5cv/drdi_train_matrix_fold',num2str(i),'.txt']); % 124x177
    A5 = load(['../data_ddt/5cv/drtar_train_matrix_fold',num2str(i),'.txt']); % 124x104
    A6 = load(['../data_ddt/5cv/ditar_train_matrix_fold',num2str(i),'.txt']); % 177x104
    A7 = load(['../data_ddt/5cv/didr_train_matrix_fold',num2str(i),'.txt']); % 177x124
    A8 = load(['../data_ddt/5cv/tardr_train_matrix_fold',num2str(i),'.txt']); % 104x124
    A9 = load(['../data_ddt/5cv/tardi_train_matrix_fold',num2str(i),'.txt']); % 104x177
    
    % 初始化大方阵
    bigMatrix = zeros(405, 405);
    
    % 定义起始索引
    startIndices = [1, 125, 302];
    
    % 填充大方阵
    bigMatrix(startIndices(1):startIndices(1)+123, startIndices(1):startIndices(1)+123) = A1;
    bigMatrix(startIndices(2):startIndices(2)+176, startIndices(2):startIndices(2)+176) = A2;
    bigMatrix(startIndices(3):startIndices(3)+103, startIndices(3):startIndices(3)+103) = A3;
    
    bigMatrix(startIndices(1):startIndices(1)+123, startIndices(2):startIndices(2)+176) = A4;
    bigMatrix(startIndices(1):startIndices(1)+123, startIndices(3):startIndices(3)+103) = A5;
    
    bigMatrix(startIndices(2):startIndices(2)+176, startIndices(3):startIndices(3)+103) = A6;
    bigMatrix(startIndices(2):startIndices(2)+176, startIndices(1):startIndices(1)+123) = A7;
    
    bigMatrix(startIndices(3):startIndices(3)+103, startIndices(1):startIndices(1)+123) = A8;
    bigMatrix(startIndices(3):startIndices(3)+103, startIndices(2):startIndices(2)+176) = A9;
    
    % 保存结果
    dlmwrite(['../data_ddt/diffusion_5cv/big_Matrix_fold',num2str(i),'.txt'], bigMatrix, 'delimiter', ' ', 'precision', '%.15g');
end
