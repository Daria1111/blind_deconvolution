clear;clc;
rng(15);
%pic = ones(10,10);
%pic(3:6, 4:7) = zeros(4,4);
%pic(:,3) = zeros(10,1);
%pic(:,8) = zeros(10,1);
pic = checkerboard(1);

%figure
%imshow(pic)
y = pic(:);
N = 64; 

L = 3;






B = randn(N,L);


alpha = randn(L,1);              
h = B*alpha; 


y = fft(y);
y_obs = diag(h) * y;

addpath 'C:\Program Files\Mosek\9.1\toolbox\R2015a'
cvx_solver mosek
cvx_begin sdp quiet
    variable u(N-1,1) complex
    variable M(L,L) hermitian
    variable Z(N,L) complex
    
    variable t
    variable aux(N,1) complex
    dual variable dual_var;
    minimize  1/2*(t*N+trace(M) )
    subject to
          [toeplitz([t; u]), Z;
            Z' , M ]>=0
         dual_var: y_obs ==  diag(Z*B.');

cvx_end

%[U,S,V] = svd(Z);

%col = U(:,5);
%col = ifft(col);
%Z = ifft(Z)./ h;
res = abs(ifft(Z) / h(1));
result = res(:,1);
figure
imshow(reshape(result, [8,8]))
err = norm(y_obs - diag(h) * Z(:,1), 'fro')
