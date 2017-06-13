%%%%%TEST COIL20%%%%%%%%%
clc,clear;


load('OBJ');
len = length(OBJ);
len = 180;
x = 1:1:len;
y(1:len) = OBJ(1:len);
plot(x,y,'r-','LineWidth',2);
    
xlabel('Number of iteration','FontSize',14');
ylabel('Objective function value','FontSize',14); 
