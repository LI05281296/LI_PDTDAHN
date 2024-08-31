for i = 1:5
    dim_drug = 100;
    dim_disease = 100;
    dim_target = 100;
    dim_imc = 50;
    interactiondrdi = load('../data_ddt/drdi_matrix.txt');
    interactiondrtar = load('../data_ddt/drtar_matrix.txt');
    interactionditar = load('../data_ddt/ditar_matrix.txt');
    interactiondrdif = load(['../data_ddt/5cv/drdi_train_matrix_fold',num2str(i),'.txt']);
    interactiondrtarf = load(['../data_ddt/5cv/drtar_train_matrix_fold',num2str(i),'.txt']);
    interactionditarf = load(['../data_ddt/5cv/ditar_train_matrix_fold',num2str(i),'.txt']);
    drug_feat = load(['../feature_ddt/d100/drug_vector_d100_fold',num2str(i),'.txt']);
    disease_feat = load(['../feature_ddt/d100/disease_vector_d100_fold',num2str(i),'.txt']);
    target_feat = load(['../feature_ddt/d100/target_vector_d100_fold',num2str(i),'.txt']);
    
    nFold = 5;
    
    Zscoredrdi = DTINet(5, nFold, interactiondrdi, interactiondrdif, drug_feat, disease_feat, dim_imc);
    
    Zscoredrtar = DTINet(5, nFold, interactiondrtar, interactiondrtarf, drug_feat, target_feat, dim_imc);
    
    Zscoreditar = DTINet(5, nFold, interactionditar, interactionditarf, disease_feat, target_feat, dim_imc);

    dlmwrite(['..\result\drdi_fold' num2str(i) '.txt'], Zscoredrdi, 'delimiter', '\t');
    dlmwrite(['..\result\drtar_fold' num2str(i) '.txt'], Zscoredrtar, 'delimiter', '\t');
    dlmwrite(['..\result\ditar_fold' num2str(i) '.txt'], Zscoreditar, 'delimiter', '\t');
     
end

