clear vars; close all;
warning('off', 'MATLAB:singularMatrix');
rng('default');

%% USER CONTROLABLE VALUES
na = 20; % number of events of type A
stdb = 2;   % std. deviation of time difference A->B
meanb = 10; % mean of time difference A->B
fracb = .9; % Fraction (number of events type A) / (number of events type B). Value has to be in [0,1]. If value is equal to one, the number of events of type A is equal to the number of events of type B.

T = 100; % time horizon. 
PLOT = false; % plot events

%% Generate events, where events of type B depend on events of type A.
% Type A events
ta = sort(T*rand([1 na])); % timestamp 

% Type B events, which are correlated with type A events
if fracb == 1
    idxa = 1:na;
else
    idxa = randperm(na, floor(na*fracb));
end
nb = numel(idxa);
tb = ta(idxa) + (meanb + stdb*randn([1 nb])); % timestamp

[TA, TB] = meshgrid(ta, tb);
delta = (TB-TA)';

%% Plot events
if PLOT
    hold on; box on;
    stem(ta, ones(1,na));
    stem(tb, .9*ones(1,nb));
    legend('Event A', 'Event B');
    ylim([0, 1.1]);
    xlabel('Time');
    hold off;
end

%% Cost function
meanZ = @(Z)1/nb*Z(:)'*delta(:);
varZ = @(Z)1/nb*sum(sum(delta.^2.*Z)) - meanZ(Z)^2;

%% Constraints
% z has to be in [0,1]
lb = zeros(na*nb,1);
ub = ones(na*nb,1);

%-- Inequality constraints
% No negative time difference (out of sequence) allowed
A = -diag(delta(:));
b = zeros(na*nb,1);
% Force that an event of type A is assigned to at most one event of
% type B, i.e., an event of type A can trigger none or one event of type B
A = [A; repmat(eye(na), [1, nb])];
b = [b; ones(na,1)];

%-- Equality constraints
% Force that an event of type B is only assigned to exactly one event of
% type A, i.e., an event of type B can only be triggerd by one event and
% not by many.
Aeq = kron(eye(nb), ones(1,na));
beq = ones(nb,1);


%% Solve optimization problem
%-- Quadratic
H = 2/nb/nb*delta(:)*delta(:)';
f = 1/nb*delta(:).^2;
z_opt_quad = quadprog(H,f,A,b,Aeq,beq,lb,ub);

Z = reshape(z_opt_quad, [na, nb]);
[~, assoc_idx_quad] = max(Z);

mean_quad = meanZ(Z);
var_quad = varZ(Z);
std_quad = sqrt(var_quad);

%-- Linear
z_opt_lin = linprog(f,A,b,Aeq,beq,lb,ub);

Z = reshape(z_opt_lin, [na, nb]);
[~, assoc_idx_lin] = max(Z);

mean_lin = meanZ(Z);
var_lin = varZ(Z);
std_lin = sqrt(var_lin);

%% Display estimates and assignments
disp('Est. mean time difference:');
disp(['- True              : ', num2str(meanb)]);
disp(['- Variance/quadratic: ', num2str(mean_quad)]);
disp(['- Linear            : ', num2str(mean_lin)]);
disp(' ');
disp('Est. std. deviation of time difference: ');
disp(['- True              : ', num2str(stdb)]);
disp(['- Variance/quadratic: ', num2str(std_quad)]);
disp(['- Linear            : ', num2str(std_lin)]);
disp(' ');
disp(['Event of type B          : ', int2str(1:nb)]);
disp(['- true assignment to A   : ', int2str(idxa)]);
disp(['- assignment to A (quad) : ', int2str(assoc_idx_quad)]);
disp(['- assignment to A (lin)  : ', int2str(assoc_idx_lin)]);
disp(' ');