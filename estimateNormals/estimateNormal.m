ptCloud = pcread('bone1.pcd'); % read from a PLY file
normals = pcnormals(ptCloud);
figure
pcshow(ptCloud);
title('Point Data');

% ½ÃÕý·½Ïò
% sensorCenter =[sum(ptCloud.XLimits)/2,sum(ptCloud.YLimits)/2,sum(ptCloud.ZLimits)/2]; 
% x = ptCloud.Location(:,1);
% y = ptCloud.Location(:,2);
% z = ptCloud.Location(:,3);
% u = normals(:,1);
% v = normals(:,2);
% w = normals(:,3);
% for k = 1 : numel(x)
%    p1 = sensorCenter - [x(k),y(k),z(k)];
%    p2 = [u(k),v(k),w(k)];
%    % Flip the normal vector if it is not pointing towards the sensor.
%    angle = atan2(norm(cross(p1,p2)),p1*p2');
%    if angle > pi/2 || angle < -pi/2
%        u(k) = -u(k);
%        v(k) = -v(k);
%        w(k) = -w(k);
%    end
% end
% normals = [u,v,w];


% file_name = 'test_32.txt';
% raw_data = load(file_name);
% KDtree = KDTreeSearcher(raw_data);
% radius = 25.0;
% min_neighbors = 8;
% normal = cell(size(raw_data,1),1);
% parfor i = 1:size(raw_data,1)
%     fprintf("it has been %d times\n",i);
%     normal{i} = estimateNormals(raw_data,KDtree,raw_data(i,:),radius,min_neighbors);
% end
% 
% normals = zeros(size(raw_data));
% for i=1:size(raw_data,1)
%     t = normal{i};
%     if size(t,1) == 3
%         normals(i,:) = t';
%     end
% end
function normal = estimateNormals(data,tree,query,radius,min_neighbors)
% ESTIMATENORMAL Given a point cloud and query point, estimate the surface 
% normal by performing an eigendecomposition of the covariance matrix created 
% from the nearest neighbors of the query point for a fixed radius.
%
%  Example: 
%       data = randn(256,3);
%       tree = KDTreeSearcher(data);
%       query = [data(1,1) data(1,2) data(1,3)];
%       radius = 1.0;
%       min_neighbors = 8;
%       normal = estimateNormal(data, tree, query, radius, min_neighbors)
%
%  Copyright (c) 2014, William J. Beksi <beksi@cs.umn.edu>
% 
 
% Find neighbors within the fixed radius 
idx = rangesearch(tree, query, radius);                                                   
idxs = idx{1};
neighbors = [data(idxs(:),1) data(idxs(:),2) data(idxs(:),3)];
 
if size(neighbors) < min_neighbors
    normal = {1};
    return;
end
 
% Compute the covariance matrix C
C = cov(neighbors);
 
% Compute the eigenvector of C
[v, lambda] = eig(C);
 
% Find the eigenvector corresponding to the minimum eigenvalue in C
[~, i] = min(diag(lambda));
 
% Normalize
normal = v(:,i) ./ norm(v(:,i));
 
end

