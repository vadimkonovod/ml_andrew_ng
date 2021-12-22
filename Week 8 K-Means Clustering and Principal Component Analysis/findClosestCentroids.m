function [idx, dists] = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
dists = zeros(size(X,1), 1);

for i = (1:size(X,1))
    min_k = 0;
    min_dist = 0;
    for j = (1:K)
        k_dist = sqrt(sumsqr(centroids(j,:) - X(i,:)));
        if min_dist == 0
            min_k = j;
            min_dist = k_dist;
        elseif k_dist < min_dist
            min_k = j;
            min_dist = k_dist;
        end
    end
    idx(i) = min_k;
    dists(i) = min_dist;
end

% =============================================================

end

