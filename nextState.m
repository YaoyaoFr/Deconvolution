function [neural,BOLD,x]=nextState(currentState,TLength,Pa)
%% Initial Parameters
if(nargin<=3)
    load('Pa.mat');
end

if(nargin<=2)
    TLength=1/Pa.Vg;
end

if(nargin<=0)
    currentState=[0;1;1;1];
end


timeSpan=[0 1/Pa.Vg];
neural=zeros(1,length(timeSpan));
BOLD=zeros(1,length(timeSpan));

%% Initial Hemodyanmic State
x0=currentState;
%% Generate Random Neural Active
neural=generateNeural(0,Pa.b);
%% Balloon Model
balloon=@(t,x)balloonModel(t,x,Pa,neural);

%% Solver Differential Equation
[t,x]=ode45(balloon,timeSpan,x0);
x=x(size(x,1),:);
% figure;
% plot(t,x);
% legend('s','f','v','q');

%% Generate BOLD Signal
V0=0.02;
k1=7*Pa.Rho;
k2=2;
k3=2*Pa.Rho-0.2;
BOLD=V0*(k1*(1.-x(:,4))+k2*(1.-x(:,4)./x(:,3))+k3*(1.-x(:,3)));
end

function e=generateNeural(timeSpan,probability)
for i=1:length(timeSpan)
    if(rand()<probability)
        e(i)=1;
    else
        e(i)=0;
    end
end
end