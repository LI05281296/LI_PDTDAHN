function [W, H] = train_mf(X, drug_feat, prot_feat, options)
    % 解析 options 字符串
    opts = strsplit(options, ' ');
    lambda = str2double(opts{find(strcmp(opts, '-l')) + 1});
    k = str2double(opts{find(strcmp(opts, '-k')) + 1});
    max_iter = str2double(opts{find(strcmp(opts, '-t')) + 1});

    d = size(drug_feat, 2);  % d 是药物特征的维度
    p = size(prot_feat, 2);  % p 是蛋白质特征的维度

    % 初始化 W 和 H
    W = rand(d, k);  % W 的维度是 d x k
    H = rand(k, p);  % H 的维度是 k x p

    for iter = 1:max_iter
        % 更新 W
        for i = 1:d
            W(i, :) = (X(i, :) * H') / (H * H' + lambda * eye(k));
        end

        % 更新 H
        for j = 1:p
            H(:, j) = (W' * W + lambda * eye(k)) \ (W' * X(:, j));
        end

        % 计算重构误差（可选）
        if mod(iter, 10) == 0
            reconstruction_error = norm(X - drug_feat * W * H * prot_feat', 'fro');
            fprintf('Iteration %d, Reconstruction error: %f\n', iter, reconstruction_error);
        end
    end
end
