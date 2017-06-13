function result = cosdis(X)
[m,n]=size(X);
A = zeros(m,m);
for i =1:m
    for j = 1:m
        A(i,j)=dot(X(i,:),X(j,:))/(norm(X(i,:))*norm(X(j,:)));
    end
end
result = A;