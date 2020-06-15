clear
load 3Dbeta10
[X,Y] = meshgrid(1:1:11,1:11);
Z = accuracy4_m;%(X,Y);% sin(X) + cos(Y);
surf(X,Y,Z)