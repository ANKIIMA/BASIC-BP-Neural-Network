close all;clc

N_train = 50:25:125;
N_test = 150-N_train;
load fisheriris

X = [ones(150,1) meas];

Y_given = zeros(size(X,1),2);
Y_given(1:50,1) = 1;
Y_given(51:100,2) = 1;
Y_given(101:150,3) = 1;

input_layer_size = size(X,2);  
hidden_layer_size = 11;
output_layer_size = size(Y_given,2);

max_iters = 1e4; %迭代次数


rho = 0.005;%学习率

cost_thres = 0.25;%损失函数阈值

train_accuracy = zeros(4,length(N_train));
test_accuracy = zeros(4,length(N_train));

sigmoid = @(z) (1.0./(1.0+exp(-z)));
sigmoidPrime = @(z) (sigmoid(z).*(1-sigmoid(z)));

for rr=1:4
    for nn=1:length(N_train)
        
        k_vals = 1:max_iters;
        J_vals = zeros(1,max_iters);
        acc_vals = zeros(1,max_iters);

        rng(10*rr)

        W1 = randn(hidden_layer_size,input_layer_size)';

        W2 = randn(output_layer_size,hidden_layer_size)';

        rp = randperm(150,N_train(nn));

        rp_comp = zeros(1,N_test(nn)); 

        kk=1;
        for ii=1:150
            if sum(ii==rp)==0
                rp_comp(kk)=ii;
                kk=kk+1;
            end
        end

        Y_train = Y_given(rp,:);

        for k=1:max_iters
            a1 = X(rp,:);

            z2 = a1*W1;

            a2 = sigmoid(z2);

            z3 = a2*W2;

            yy = sigmoid(z3);

            del_out = -(Y_train-yy).*sigmoidPrime(z3);

            del2 = (del_out*W2').*sigmoidPrime(z2);

            W2_pd = a2'*del_out;

            W1_pd = a1'*del2;

            W1 = W1 - rho*W1_pd;
            W2 = W2 - rho*W2_pd;

            J = norm(Y_train(:)-yy(:))/sqrt(length(yy(:)));

            [~,m1] = max(yy,[],2);
            [~,m2] = max(Y_train,[],2);
            train_accuracy(rr,nn) = mean(m1==m2)*100;
  
            J_vals(k) = J;
            acc_vals(k) = train_accuracy(rr,nn);            
            
            if train_accuracy(rr,nn)==100 && J<cost_thres
                k_vals(k+1:end) = [];
                J_vals(k+1:end) = [];
                acc_vals(k+1:end) = [];
                break
            end

        end

         plot(k_vals,J_vals)  
         xlabel('更新次数')
         ylabel('误差')         

        %% 测试

        Y_test_given = Y_given(rp_comp,:);

        a1 = X(rp_comp,:);

        z2 = a1*W1;

        a2 = sigmoid(z2);

        z3 = a2*W2;

        yy = sigmoid(z3);

        yy = yy./sum(yy,2);

        [~,c1] = max(yy,[],2);
        [~,c2] = max(Y_test_given,[],2);
        test_accuracy(rr,nn) = mean(c1==c2)*100;

    end
    
end

train_accuracy_avg = mean(train_accuracy,1)
test_accuracy_avg = mean(test_accuracy,1)









