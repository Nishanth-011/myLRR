%%%%%TEST COIL20%%%%%%%%%
clear;
%load COIL20
%load Yale_32x32;
load YaleBext_3232
nn = 2;
<<<<<<< HEAD
fea = fea(1:2414,:);
gnd = gnd(1:2414);
=======
fea = fea(1:64*nn,:);
gnd = gnd(1:64*nn);
>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
folder_now = pwd;
addpath([folder_now, '\funs']);
%% reduce demension by PCA
% options = [];
% options.PCARatio = 0.98;
% %options.ReducedDim = 60;
% [eigvector,eigvalue,meanData,new_data] = PCA(fea,opt ions);
% fea=new_data;
% %%
samp_num = size(fea,1);
nnClass = length(unique(gnd));  % The number of classes;
num_Class=[];
for i=1:nnClass
  num_Class=[num_Class length(find(gnd==i))]; %The number of samples of each class
end

fea =  NormalizeFea(fea);

test = fea;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  fea_num = 30;
%  fea_gnd = zeros(fea_num*nnClass, 1);
% for  j=1:nnClass
%     idx=find(gnd==j);
%     rand_idx = randperm(num_Class(j)); 
%     test = [test; fea(idx(rand_idx(1:fea_num)),:)];
%     fea_gnd((j-1)*fea_num+1:j*fea_num) = j;
% end
% test =  NormalizeFea(test);
% samp_num = fea_num*nnClass;
%%%%%%%%%%%%%%%%%%%%%
runtimes = 10;
gama_1 = 50;
gama_2 = 11;
<<<<<<< HEAD
sele = 13;
=======
sele = 6;
>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
minU0 = 1e-12;
maxU0 = 1e5;  
    %[A OBJ] = LRSA(test', gama_1, gama_2);
    %[A OBJ] = LRSA1(test', 50, 11, 2);
<<<<<<< HEAD
    %[A OBJ] = LRSA2(test', 50, 11, 4); 
    [A OBJ] = LRSA3(test', 50, 11,3,64);   
=======
    [A OBJ] = LRSA2(test', 50, 11, 4); 
    %[A OBJ] = LRSA3(test', 50, 11,5,nn*11);
    
>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
    A = NormalizeFea(A); 
    A = A + 0.0000001*ones(size(A));
    AG = A;
    save('AG','AG');
    save('OBJ','OBJ');
    W = A;
    D = diag(sum(A));
<<<<<<< HEAD
    for r=1:runtimes
%---------------------------------------------------------------   
=======

    for r=1:runtimes
%---------------------------------------------------------------   

>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
        Y = zeros(samp_num, nnClass);
        cLab = zeros(samp_num, nnClass);
        FF = zeros(samp_num, nnClass);
        TestF = ones(samp_num, nnClass);
        U0 = zeros(samp_num, samp_num);
        Umin = minU0*ones(samp_num, samp_num);
<<<<<<< HEAD
=======

>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
        for  j=1:nnClass
            idx=find(gnd==j);
            cLab(idx, j) = 1;
            randIdx=randperm(num_Class(j)); %randIdx create m random number, m is the size of idx.
            %randIdx = 1:sele;
            Y(idx(randIdx(1:sele)),j) = 1;
            TestF(idx(randIdx(1:sele)),:) = 0;      
            for s = 1:sele
                U0(idx(randIdx(s)),idx(randIdx(s))) = maxU0;
            end                
        end
<<<<<<< HEAD
=======

>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
        F = (D+U0-W+Umin)\U0*Y;
        [maxF, idF] = max(F,[],2);
        for j = 1:samp_num
            FF(j,idF(j)) = 1;
        end
<<<<<<< HEAD
=======

>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
        recogNum = sum(sum((cLab.*FF).*TestF));
        testNum = samp_num-sele*nnClass;
        ratio = double(recogNum)/testNum;
        rate(r) = ratio;
    end
    %max(rate)
    mean(rate)
<<<<<<< HEAD
    mean(sele)
    std(rate)
%---------------------------------------------------------------
=======
    std(rate)
%---------------------------------------------------------------


>>>>>>> 2cba506f9191543c7965a14801e5dca242438412
