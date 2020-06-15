function [TPR,FPR,Precision,Recall,F1] = performanceIndexs(u,v)
%performanceIndexs perform the ARI (Adjusted Rand Index),TPR and FPR 
% between two clustering u and v.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2017 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
% All rights reserved.
% this software should be used, reproduced and modified only for informational and nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt
% (included in this package) and online at
% http://www.grip.unina.it/download/LICENSE_OPEN.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n=length(u);
ku=max(u);
kv=max(v);
m=zeros(ku,kv);
for i=1:n
    m(u(i),v(i))=m(u(i),v(i))+1;
end
mu=sum(m,2);
mv=sum(m,1);

a=0;
for i=1:ku
    for j=1:kv
        if m(i,j)>1
            a=a+nchoosek(m(i,j),2);
        end
    end
end

b1=0;
b2=0;
for i=1:ku
    if mu(i)>1
        b1=b1+nchoosek(mu(i),2);
    end
end
for i=1:kv
    if mv(i)>1
        b2=b2+nchoosek(mv(i),2);
    end
end

c=nchoosek(n,2);

TP = a;
FP = b1-TP;
FN = b2-TP;
TN = c-TP-FP-FN;

%ARI =(a-b1*b2/c)/(0.5*(b1+b2)-b1*b2/c);
TPR = TP/(TP+FN);
FPR = FP/(FP+TN);
Precision = TP/(TP+FP);
Recall = TPR;
F1 = 2*Precision*Recall/(Recall+Precision);

%m = (TP+FP)*(TP+FN)+(FN+TN)*(FP+TN);
%ARI = (c*(TP+TN)-m)/(c^2-m);







