clear; clc;
rng(1);
pic = imread('popug.jfif');

pic = rgb2gray(pic);
pic = double(pic);

m = pic(:);

K = 30;
L = 10;
N = 30000;


dims = randi(L,[int64(L/4),1]);
values = randn([int64(L/4),1]);
C = sparse(dims,dims,values,L,N);
%C = randn(L,N);
%C = full(C.*sp);
nzC = nnz(C)/numel(C);


h = randn(K,1);
h = sparse(h);
dims = randi(L,[int64(L/4),1]);
values = randn([int64(L/4),1]);
B = sparse(dims,dims,values,L,K);
%B = randn(L,K);
%B = full(B.*sp);
nzB = nnz(B)/numel(B);



X = m(1) * h;
for iter = 2 : N
    elem = m(iter) * h;
    X = sparse(cat(2,X,elem));
end

A = circulant(C(:,1)) * B;
for iter = 2 : N
    elem = circulant(C(:,iter)) * B;
    A = sparse(cat(1,A,elem));
end

nz = nnz(A)/numel(A);

Az  = transpose(A);
y = A * X;
nz


lambda = 100;

[Xk,E, obj, err, iter] = lrr_sparse(y,A,lambda);

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

result = reshape(uint8(res),[200,150]);
figure
imshow(result)

