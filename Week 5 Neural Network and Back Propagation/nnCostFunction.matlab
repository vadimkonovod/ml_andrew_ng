function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%DIMENSIONS: 
%   theta1 = hidden_layer_size * (input_layer_size + 1) = 25 * 401
%   theta2 = output_layear_size * (hidden_layer_size + 1) = 10 * 26
%   X      = m x (n + 1) = 5000 * 401
%   y      = m x 1 = 5000 * 1
%   grad   = 
%   J      = Scalar

a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * Theta1'); % 5000 * 25
a2 = [ones(m, 1) a2]; % 5000 * 26
h = sigmoid(a2 * Theta2'); % 5000 * 10

% convert labels to binary array
y_labeled = zeros(m, num_labels); % 5000 * 10

for i = 1:m
    if y(i) == 0
        y_labeled(i, num_labels) = 1;
    else
        y_labeled(i, y(i)) = 1;
    end
end

% calculate cost funcation

% matrix multiplication doesn't work
% J = (1/m) * sum(sum(log(h)' * (-y_labeled) - log(1-h)' * (1-y_labeled)));
% but 'times' (https://www.mathworks.com/help/matlab/ref/times.html) works well 
J = (1/m) * sum(sum(-y_labeled .* log(h) - (1-y_labeled) .* log(1-h)));

% find regularization
reg_th_1 = sum(sumsqr(Theta1(:, [2:end])));
reg_th_2 = sum(sumsqr(Theta2(:, [2:end])));
regularization = lambda * (reg_th_1 + reg_th_2) / (2*m);
J = J + regularization;

%DIMENSIONS: 
%   Theta1 = hidden_layer_size * (input_layer_size + 1) = 25 * 401
%   Theta2 = output_layear_size * (hidden_layer_size + 1) = 10 * 26
%   X      = m x (n + 1) = 5000 * 401
%   y      = m x 1 = 5000 * 1

% back propagation: find gradients
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

for i = 1:m
    % forward prop
    a1_i = [1; X(i,:)']; % 401 * 1
    z2_i = Theta1 * a1_i; % 25 * 1
    a2_i = sigmoid(z2_i); % 25 * 1
    a2_i = [1; a2_i]; % 26 * 1
    z3_i = Theta2 * a2_i; % 10 * 1;
    h = sigmoid(z3_i); % 10 * 1;

    % convert y to 'logical array'
    y_lbld = zeros(1, num_labels); % 1 * 10
    if y(i) == 0
        y_lbld(1, num_labels) = 1;
    else
        y_lbld(1, y(i)) = 1;
    end
    
    % calculate delta_3
    delta_3 = h - y_lbld'; % 10 * 1

    % calculate delta_2
    z2_i = [1; z2_i];
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2_i); % 26 * 1
    delta_2 = delta_2(2:end); % 25 * 1

    DELTA1 = DELTA1 + delta_2 * a1_i'; % 25 * 401
    DELTA2 = DELTA2 + delta_3 * a2_i'; % 10 * 26
end

Theta1_grad = DELTA1 ./ m; % 25 * 401
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);

Theta2_grad = DELTA2 ./ m; % 10 * 26
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
