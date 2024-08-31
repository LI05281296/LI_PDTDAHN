function Zscore = DTINet(seed, nFold, interaction, interactionf, drug_feat, prot_feat, dim_imc)
	rng(seed);

	for foldID = 1 : nFold
        train_posIdx = find(interactionf);
        % 找到负样本的索引
        % 负样本是训练矩阵中为零且在原始交互矩阵中也是零的索引
        Pnoint = find(~interaction);
        train_negIdx_candidates = Pnoint(ismember(Pnoint, find(interactionf == 0)));

         % 随机选择和正样本数量相同的负样本
        train_negIdx = train_negIdx_candidates(randperm(length(train_negIdx_candidates), length(train_posIdx)));
        
        % 创建标签向量
        Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];
        
        % 合并正负样本的索引
        train_idx = [train_posIdx; train_negIdx];
        
        % 输出标签统计信息
        fprintf('Train data: %d positives, %d negatives\n', sum(Ytrain == 1), sum(Ytrain == 0));
        
        % 输出原始训练索引
        fprintf('Train indices restored, total: %d\n', length(train_idx));

		[I, J] = ind2sub(size(interaction), train_idx);
		Xtrain = sparse(I, J, Ytrain, size(interaction, 1), size(interaction, 2));
        [W, H] = nnmf(Xtrain, dim_imc);
		%[W, H] = train_mf(Xtrain, sparse(drug_feat), sparse(prot_feat), ...
		%				[' -l ' num2str(1) ' -k ' num2str(dim_imc) ' -t 10' ' -s ' num2str(5)]); 
		Zscore =  (W * H + drug_feat * prot_feat');

end
