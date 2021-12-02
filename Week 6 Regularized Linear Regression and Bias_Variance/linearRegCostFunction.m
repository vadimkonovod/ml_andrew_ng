function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% DIMESIONS
% X = m * n = 12 * 2
% Y = m * 1 = 12 * 1
% theta = n * 1 = 2 * 1

J = (sumsqr(X * theta - y) + lambda * sumsqr(theta(2:end))) / (2 * m) ;

grad = X' * (X * theta - y) / m;
grad_0 = grad(1);
grad = grad + theta * lambda / m;
grad(1) = grad_0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
