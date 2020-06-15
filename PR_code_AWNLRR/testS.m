%用来测试S有什么特别的性质
Sq = sqrt(S);
U = fea*Z;
weight = Sq.*(fea - fea*Z);
% imshow((Sq(:,73:72*3)*100-3)*5)