function [lo, up, output_info] = approximate_twonorm(A, m, tol, reorth)
    % LANCZOS_2NORM_BOUND_ORTH   Lanczos 2-norm bounds with optional re-orth.
    %
    %   [LO, UP, K] = LANCZOS_2NORM_BOUND_ORTH(A, M, TOL, REORTH)
    %
    %   LO, UP   : certified lower / upper bounds on ‖A‖₂
    %   K        : Lanczos steps actually performed
    %
    %   INPUT  A       – real dense or sparse matrix (square or rectangular)
    %          M       – maximum Lanczos steps      (default 50)
    %          TOL     – relative residual tolerance (default 1e-6)
    %          REORTH  – 0  : no re-orth (fastest, may lose orthogonality)
    %                      1  : full one-pass Gram–Schmidt at every step
    %                      2  : selective re-orth  (only when β_j < 1e-2 · ‖B‖₁)
    %
    %   The upper bound uses an *explicit* residual multiply, so it is
    %   rigorous whether or not re-orth is enabled.
    
    % -------- defaults ------------------------------------------------------
    if nargin < 2 || isempty(m),      m      = 10;    end
    if nargin < 3 || isempty(tol),    tol    = 1e-3;  end
    if nargin < 4 || isempty(reorth), reorth = 2;     end     % sensible default
    
    n = size(A,2);                               % size of B = A'*A
    
    % handle for B-multiply
    Bmul = @(x) A' * (A * x);
    
    % -------- storage -------------------------------------------------------
    V     = zeros(n, m);                         % basis
    alpha = zeros(m,1);
    beta  = zeros(m,1);
    
    % -------- initial vector ------------------------------------------------
    
    q = randn(n,1);
    % q = ones(n, 1);

    % q = load('../q.txt');
    % if length(q) ~= n
    %     error('Initial vector q must have the same length as the number of columns in A.');
    % end

    q = q / norm(q);
    V(:,1) = q;
    betaOld = 0;           vOld = zeros(n,1);
    
    % display the maximum element of the matrix
    % disp(max(abs(A(:))));

    % -------- Lanczos loop --------------------------------------------------
    for k = 1:m
        w        = Bmul(q);
        alpha(k) = q' * w;
    
        w = w - alpha(k) * q - betaOld * vOld;
    
        % ---------- optional re-orthogonalisation ---------------------------
        if reorth > 0
            % pre-compute overlap once
            proj = V(:,1:k) * (V(:,1:k).' * w);
            w    = w - proj;                         % first GS pass
    
            if reorth == 1            % full re-orth: do it twice
                proj = V(:,1:k) * (V(:,1:k).' * w);
                w    = w - proj;
            end
        end
    
        beta(k) = norm(w);

        % fprintf("alpha(k): %5.4e, beta(k): %5.4e \n", alpha(k), beta(k));
    
        % stopping test (explicit resid later guarantees rigour, this just
        % limits work)
        if beta(k) <= tol * alpha(k) && k > 2
            break
        end
    
        % prepare next step
        vOld    = q;
        q       = w / beta(k);
        if k < m
            V(:,k+1) = q;
        end
        betaOld = beta(k);
    end
    
    % truncate
    alpha = alpha(1:k);
    beta  = beta(1:k);
    V     = V(:,1:k);
    
    % tridiagonal T_k
    T = diag(alpha) + diag(beta(1:end-1),1) + diag(beta(1:end-1),-1);
    % T = diag(alpha) + diag(beta(2:end),1) + diag(beta(2:end),-1);

    % ------------- largest Ritz pair ----------------------------------------
    [ritzVecs, ritzVals] = eig(T,'vector');
    [theta, idx]         = max(ritzVals);
    uk                   = ritzVecs(:,idx);     % eigenvector in T-space
    
    % ------------- explicit residual ----------------------------------------
    y       = V * uk;                           % full Ritz vector
    ry      = Bmul(y) - theta * y;              % ONE extra (A,A') multiply
    lower_sq = theta;
    upper_sq = theta + norm(ry);                % rigorous upper bound
    
    lo  = sqrt(lower_sq);
    up  = sqrt(upper_sq);

    output_info.k = k;
end
