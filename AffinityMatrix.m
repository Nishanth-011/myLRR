
function [affinity] = AffinityMatrix(data)

sigma = 1;
for i=1:size(data,1)
    a=ones(size(data,1),1)*data(i,:)-data;
    for j=1:size(data,1)
        dist=norm(a(j,:));
        affinity(i,j)= exp(-dist/(2*sigma^2)); 
    end
end
end