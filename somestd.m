clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Results/newrevise/');
% load sORL_32x32fast
%ARI	NMI	ACC	Precision	Recall	F-score
methodname = {'kmeans','njw','fast','lrr','latlrr','nnllrr','lrradpn','awlrr','ours'};
% dataname = 'sCOIL20';
dataname = 'sYale_32x32';
% dataname = 'sORL_32x32';
% dataname = 'sUmist';
% dataname = 'smnist_all';
% dataname = 'sCCC40';
% dataname = 'sPD100';
AAAresult=cell(9,5);
for time = 1:9
    name = strcat(dataname,methodname(time));
    load(name{1})
    score = [std(aARI);std(aNMI);std(aACC);std(aPrecision);std(aF1)];
    scorem = [amARI;amNMI;amACC;amPrecision;amF1];
    for i = 1:5
        a = num2str(score(i));
        if length(a)>=5
            a = a(1:5);
        else
            a = strcat(a,'000000000');
            a = a(1:5);
        end
        b = num2str(scorem(i));
        if length(b)>=5
            b = b(1:5);
        else
            b = strcat(b,'000000000');
            b = b(1:5);
        end
        AAAresult(time,i) = {strcat(b,'=',a)};
    end
end