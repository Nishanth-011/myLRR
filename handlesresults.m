% clear;
% addpath('Datasets/');
% addpath('Functions/');
% addpath('Results/newrevise/');
% % load sORL_32x32fast
% %ARI	NMI	ACC	Precision	Recall	F-score
% methodname = {'lrr','latlrr','nnllrr','lrradp','ours'};
% dataname = 'ssCOIL20';
% AAAresult=cell(15,7);
% for Ttime = 1:5
%     name = strcat(dataname,methodname(Ttime));
%     load(name{1})
%     score = [std(aACC,[],2)';std(aPrecision,[],2)';std(aF1,[],2)'];
%     scorem = [amACC';amPrecision';amF1'];
%     for j = 1:3
%         for i = 1:7
%             a = num2str(score(j,i));
%             if length(a)>=5
%                 a = a(1:5);
%             else
%                 a = strcat(a,'000000000');
%                 a = a(1:5);
%             end
%             b = num2str(scorem(j,i));
%             if length(b)>=5
%                 b = b(1:5);
%             else
%                 b = strcat(b,'000000000');
%                 b = b(1:5);
%             end
%             AAAresult(Ttime*3 -3+j,i) = {strcat(b,'=',a)};
%         end
%     end
% end


clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Results/newrevise/');
% load sORL_32x32fast
%ARI	NMI	ACC	Precision	Recall	F-score
methodname = {'lrr','latlrr','nnllrr','lrradp','awlrr','ours'};
% dataname = 'ssCOIL20';
% dataname = 'ssYale_64x64';
% dataname = 'ssORL_32x32';
% dataname = 'ssUmist';
% dataname = 'ssmnist_all';
% dataname = 'ssCCC40';
dataname = 'ssPD100';
AAAresult=cell(21,6);
for Ttime = 1:6
    name = strcat(dataname,methodname(Ttime));
    load(name{1})
    score = [std(aACC,[],2)';std(aPrecision,[],2)';std(aF1,[],2)'];
    scorem = [amACC';amTPR';amF1'];
    score = score';
    scorem = scorem';
    for j = 1:3
        for i = 1:7
            a = num2str(score(i,j));
            if length(a)>=5
                a = a(1:5);
            else
                a = strcat(a,'000000000');
                a = a(1:5);
            end
            b = num2str(scorem(i,j));
            if length(b)>=5
                b = b(1:5);
            else
                b = strcat(b,'000000000');
                b = b(1:5);
            end
            AAAresult(i*3-3+j,Ttime) = {strcat(b,'=',a)};
        end
    end
end