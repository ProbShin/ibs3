%%
% BLOCK LU_factorization, no pivoting
clc
clear
A = mmread('tinyA_dd.mtx');
b = mmread('tinyb_dd.mtx');
[m,n] = size(A);
I=eye(n/8);

%make a copy to check the residual at the end
Acopy = A;
bcopy = b;

%initialize the Lower triangular matrix
L = zeros(n,n);

%work on block column 'i'
for i = 1:n/8
    br1 = (n/8)*(i-1)+1;
    br2 = (n/8)*i;
    [Lfac,Ufac] = lu(A(br1:br2,br1:br2));
    L(br1:br2,br1:br2) = I;
   
    for var1 = (i+1):(n/8) %update block row A_(var1,var2)
        row1 = (n/8)*(var1-1) + 1;
        row2 = (n/8)*var1;
        %G = A(row1:row2,br1:br2)*inv(Ufac)*inv(Lfac);
        G = A(row1:row2,br1:br2)*inv(Ufac)*inv(Lfac);
        %update L
        L(row1:row2,br1:br2) = G;
         
        %update U
        for var2 = i+1:n/8
            col1 = (n/8)*(var2-1)+1;
            col2 = (n/8)*var2;
            A(row1:row2,col1:col2) = -G*A(br1:br2,col1:col2) + A(row1:row2,col1:col2);
        end
        A(row1:row2,br1:br2) = 0;
    end
end

U = A;

%LUx = b
y = L\b; %solve Ly = b
x = U\y; %solve Ux = y

residual = norm(Acopy*x - bcopy)/norm(bcopy)

