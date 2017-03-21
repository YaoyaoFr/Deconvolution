function [neural,BOLD,finalState]=Simulation_old(state,TLength,batchSize,Pa)
%% Initial Parameters
if(nargin<=3)
    load('Pa.mat');
end

if(nargin<=2)
    batchSize=128;
end

if(nargin<=1)
    TLength=32;
end

if(nargin<=0)
    state=[zeros(1,batchSize);ones(1,batchSize);ones(1,batchSize);ones(1,batchSize)];
end

if(size(state,2)~=batchSize)
    error('State dimension is error!');
end

timeSpan=0:1/Pa.Vg:TLength;
neural=zeros(batchSize,length(timeSpan));
BOLD=zeros(batchSize,length(timeSpan));
finalState=zeros(size(state,1),batchSize);

for i=1:batchSize
    %% Initial Hemodyanmic State
    x0=state(:,i);
    
    %% Generate Random Neural Active
    e=generateNeural(timeSpan,Pa.b);
    neural(i,:)=e;
    
    %% Balloon Model
    balloon=@(t,x)balloonModel(t,x,Pa,e);
    
    %% Solver Differential Equation
    [t,x]=ode45(balloon,timeSpan,x0);
    
    % figure;
    % plot(t,x);
    % legend('s','f','v','q');
    
    %% Generate BOLD Signal
    V0=0.02;
    k1=7*Pa.Rho;
    k2=2;
    k3=2*Pa.Rho-0.2;
    % for i=1:size(x,1)
    %     y(i)=V0*(k1*(1-x(i,4))+k2*(1-x(i,4)/x(i,3))+k3*(1-x(i,3)));
    % end
    
    y=V0*(k1*(1.-x(:,4))+k2*(1.-x(:,4)./x(:,3))+k3*(1.-x(:,3)));
    BOLD(i,:)=y*20;
    finalState(:,i)=x(length(timeSpan),:);
end
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