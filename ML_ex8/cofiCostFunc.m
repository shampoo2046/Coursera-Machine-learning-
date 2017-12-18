function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
sum =0; 
% M =(X*Theta'-Y).^2;
% Mx = M(R);
% for i=1:length(Mx);
%    sum = sum+Mx(i);    
% end

for i= 1: num_users
    for j=1: num_movies
     if R(j,i)==1;
         sum = sum+(Theta(i,:)*X(j,:)'-Y(j,i))^2;
     end;
    end;
end;

% t1 = lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));
t1 = Theta.^2;
t2 = X.^2;
s1=0;
s2=0;
for i=1: num_users
    for j=1:num_features
        s1 =s1 + t1(i,j);
    end;
end;
for i=1: num_movies
    for j=1:num_features
        s2 =s2 + t2(i,j);
    end;
end;
J = 1/2*sum +lambda/2*(s1+s2);

% M1 =(X*Theta'-Y);
for i= 1: num_movies
    idx = find(R(i, :)==1);
    Theta_temp = Theta(idx,:);
    Ytemp = Y(i,idx);
    X_grad(i,:)=(X(i,:)*Theta_temp'-Ytemp)*Theta_temp+lambda*X(i,:);
end;

for i= 1: num_users
    idx = find(R(:, i)==1);
    X_temp = X(idx,:);
    Ytemp = Y(idx,i);
    Theta_grad(i,:)=(X_temp*Theta(i,:)'-Ytemp)'*X_temp+lambda*Theta(i,:);
end;
% idx = find(R(:,:)==1);
% for k= 1: num_features
%     for i= 1: num_users
%         for j=1: num_movies
%          if R(j,i)==1;
%               mm(j,i,k)=(Theta(i,:)*X(j,:)'-Y(j,i))*X(j,k);
%          end;
%         end;
%        Theta_grad(i,k) = Theta_grad(i,k)+mm(j,i,k); 
%     end;
% end;       
%     








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
