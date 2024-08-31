
maxiter = 20;
restartProb = 0.50;
dim = 100;
for i = 1:5
    Nets = {['big_Matrix_fold',num2str(i)]};    
    tic
    X = DCA(Nets, dim, restartProb, maxiter);
    toc
    dr = X(1:124,:);
    di = X(125:301,:);
    tar = X(302:end,:);

    dlmwrite(['../feature_ddt/d100/drug_vector_d100_fold', num2str(i), '.txt'], dr, '\t');
    dlmwrite(['../feature_ddt/d100/disease_vector_d100_fold', num2str(i), '.txt'], di, '\t');
    dlmwrite(['../feature_ddt/d100/target_vector_d100_fold', num2str(i), '.txt'], tar, '\t');
end