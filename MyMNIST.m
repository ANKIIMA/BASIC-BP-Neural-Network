clear all;
close all;
clc;

load mnist_uint8;
er = [];
for i = 1:1
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

mu=mean(train_x);    
sigma=max(std(train_x),eps);
train_x=bsxfun(@minus,train_x,mu);   
train_x=bsxfun(@rdivide,train_x,sigma);   

test_x=bsxfun(@minus,test_x,mu);
test_x=bsxfun(@rdivide,test_x,sigma);

arc = [784 300 100 10]; 
n=numel(arc);

W = cell(1,n-1); 
for i=2:n
    W{i-1} = (rand(arc(i),arc(i-1)+1)-0.5) * 8 *sqrt(6 / (arc(i)+arc(i-1)));
end

learningRate = 2; 
numepochs = 1;   
batchsize = 100; 

m = size(train_x, 1);
numbatches = m / batchsize;

%% 训练
L = zeros(numepochs*numbatches,1);
ll=1;
for i = 1 : numepochs
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

       %% 正向传播
        mm = size(batch_x,1);
        x = [ones(mm,1) batch_x];
        a{1} = x;
        %隐藏层用tanh
        for ii = 2 : n-1
            a{ii} = 1./(1+exp(-(a{ii - 1} * W{ii - 1}')));   
            a{ii} = [ones(mm,1) a{ii}];
        end
        %最后一层使用sigmoid
        a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));
        e = batch_y - a{n};
        L(ll) = 1/size(e,2) * sum(sum(e.^2)) / mm; 
        ll=ll+1;
       %% 反向传播
        d{n} = ((a{n}.*(1 - a{n})) .* ((a{n} - batch_y) * 1/n * 2));
        for ii = (n - 1) : -1 : 2
            if ii+1==n    
                d{ii} = d{ii + 1} * W{ii}; 
            else 
                d{ii} = d{ii + 1}(:,2:end) * W{ii};
            end          
        end
         
        for ii = 1 : n-1
            if ii + 1 == n
                dW{ii} = (d{ii + 1}' * a{ii}) / size(d{ii + 1}, 1);
            else
                dW{ii} = (d{ii + 1}(:,2:end)' * a{ii}) / size(d{ii + 1}, 1);      
            end
        end
         
       %% 更新参数
        for ii = 1 : n - 1       
            W{ii} = W{ii} - learningRate*dW{ii};
        end
              
    end
end

%% 测试
mm = size(test_x,1);
x = [ones(mm,1) test_x];
a{1} = x;
for ii = 2 : n-1    
    a{ii} = 1./(1+exp(-a{ii - 1} * W{ii - 1}'));  
    a{ii} = [ones(mm,1) a{ii}];
end
a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

[~, i] = max(a{end},[],2);
labels = i;          
[~, expected] = max(test_y,[],2);
bad = find(labels ~= expected);   
er = [er, numel(bad) / size(x, 1)];    
end
mean(er)
std(er,1)
plot(L);