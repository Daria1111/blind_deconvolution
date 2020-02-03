clear; clc;
rng(1);
pic = checkerboard(1);
figure
imshow(pic)
m = pic(:);

K = 50;
L = 100;
N = 64;
%seed = rng(10, 'philox')
dims = randi(N,[int64(N/3),1]);
odin = ones([int64(N/3),1]);
sp = sparse(dims,dims,odin,L,N);
C = randn(L,N);
C = full(C.*sp);
nzC = nnz(C)/numel(C);
%rng(seed)

h = randn(K,1);
dims = randi(K,[int64(K/2),1]);
odin = ones([int64(K/2),1]);
sp = sparse(dims,dims,odin,L,K);
B = randn(L,K);
B = full(B.*sp);
nzB = nnz(B)/numel(B);
X = m(1) * h;
for iter = 2 : N
    elem = m(iter) * h;
    X = cat(2,X,elem);
end

A = circulant(C(:,1)) * B;
for iter = 2 : N
    elem = circulant(C(:,iter)) * B;
    A = cat(1,A,elem);
end

nz = nnz(A)/numel(A);

Az  = transpose(A);
y = A * X;



b = 100;
a = 1;
delta = 0.001;
k_max = 5000;
epsilon = 1;

Xk = zeros(K,N);
Y = zeros(K,N);
%Y = Az * y * b / (norm(y) * delta );
metrics = [];

for iter = 1 : k_max
    [U,S,V] = svd(Y);
    tau = b * exp(-a * iter);
    
    thres = (S > tau);
    S = S.*thres;
    Xk1 = Xk;
    
    Xk = U * S * V';
    Y = Y + delta * Az*(y - A * Xk);
  
    %col1 = randi(N);
    
    
              
    metric = norm(y - A*Xk,'fro')
    metrics(iter) = metric;
    if metric < epsilon
        iter
        break
    end

end


res = zeros(1,N);

for iter = 1 : N 
    elem = Xk(:,iter);
   
    mi = elem./ h;
    for i = 1 : K
        if mi(i) ~= 0
            res(iter) = mi(i);
            break
        else
            res(iter) = mi(1);
        end
    end
   
    
   
end

result = reshape(res,[8,8]);
figure
imshow(result)

figure
plot(metrics)

xlabel('Iteration')
ylabel('||AX_k-y||_F')
