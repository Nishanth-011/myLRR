% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function distance = L2_distance_1(a,b,k)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b

maxd = 1e6;
n = size(a,2);


if nargin < 3
    k = 10;
end;

if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 

d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);

[Y I] = sort(d,1);
ds = maxd*ones(n);
for j = 1:n
    ds(I(1:k,j),j) = d(I(1:k,j),j);
end

for ii = 1:n
    for jj = (ii+1):n
        ds(ii,jj) = min(ds(ii,jj), ds(jj,ii));
        ds(jj,ii) = ds(ii,jj);
    end
end

isUpdate = 1;

while isUpdate
    isUpdate = 0;
    for ii = 1:n
        for jj = (ii+1):n
%
            for kk = (ii+1):n
   
                
                if ds(ii,kk)+d(kk,jj) < ds(ii,jj)
                   ds(ii,jj) = ds(ii,kk)+ds(kk,jj);
                   ds(jj,ii) = ds(ii,jj);
                   isUpdate = 1;
                end                
            end
        end
    end
end

distance = ds;








% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end
