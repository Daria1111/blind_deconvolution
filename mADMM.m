clear; clc;
rng(1);
pic = checkerboard(1);
figure
imshow(pic)
m = pic(:);

K = 50;
L = 100;
N = 64;

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


lambda = 100;

[Xk,E, obj, errs, iter] = lrr(y,A,lambda);

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
plot(errs)

xlabel('Iteration')
ylabel('||AX_k-y||_F')

