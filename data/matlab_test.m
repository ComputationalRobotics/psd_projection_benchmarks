fid = fopen('./bin/parter-5000.bin', 'rb');
n = 5000;
A = fread(fid, [n, n], 'double');  % Reads column-wise
fclose(fid);

% A = 0.5 * (A + A');
% d = eig(A);
% fprintf("d_min: %3.2e, d_max: %3.2e \n", min(d), max(d));

for iter = 1: 100
    [lo, up, info] = approximate_twonorm(A, 20);
    disp(lo);
end

% x = svds(A, 1, 'largest');
% disp(x);

% x = eigs(A, 1);
% disp(x);

% [P, D] = eig(A);
% % extract the diagonal of D
% d = diag(D);

% % disp the maximum value of D in absolute value
% disp(max(abs(d)));

% % save d to a txt file
% writematrix(d, 'clement-10000-eigenvalues.txt', 'Delimiter', 'tab');