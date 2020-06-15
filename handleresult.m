clear;
addpath('result/');
%0----kmeans 
%1----NJW
%2----NSLLR
%3----LRRADP
%4----LRRHWAP
%5----Fast

% load reviseORL_32x32g_2p
% load reviseUmist5
% load reviseCOIL205
% load reviseYaleB_32323
% load revisemnist_allg2p
% load('reviseCCC40g_2p.mat')
% load reviseUSPSfug_2p
load revisePD100g_2p
ACC = accuracy0_m';
ACC = [ACC,accuracy1_m'];
ACC = [ACC,accuracy5_m'];
ACC = [ACC,accuracy2_m'];
ACC = [ACC,accuracy3_m'];
ACC = [ACC,accuracy4_m'];
if exist('NMI0_m','var')
    NMI = NMI0_m';
    NMI = [NMI,NMI1_m'];
    NMI = [NMI,NMI5_m'];
    NMI = [NMI,NMI2_m'];
    NMI = [NMI,NMI3_m'];
    NMI = [NMI,NMI4_m'];
else
    NMI = NMI0';
    NMI = [NMI,NMI1'];
    NMI = [NMI,NMI5'];
    NMI = [NMI,NMI2'];
    NMI = [NMI,NMI3'];
    NMI = [NMI,NMI4'];
end
result = [];
for i = 1:size(NMI, 1)
    result = [result;ACC(i,:);NMI(i,:)];
end