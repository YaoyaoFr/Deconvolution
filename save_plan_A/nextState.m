function [previousNeural,previousState,currentState,currentBOLD]=nextState(previousState,previousNeural,batchSize,Pa)
%% Initial Parameters
if(nargin<=3)
    load('Pa.mat');
end

%% Initial Batch Size
if(nargin<=2)
    batchSize=128;
end

%% Generate Random Neural Active
if(nargin<=1)
    previousNeural=generateNeural(batchSize,Pa.b);
end

%% Initial previous State
if(nargin<=0)
    previousState=[zeros(batchSize,1),ones(batchSize,3)];
end

%% Initial Output Matrix
currentBOLD=zeros(batchSize,1);
currentState=zeros(batchSize,4);

timeSpan=[0 1/Pa.Vg];

for i=1:batchSize
    %% Initial Hemodyanmic State
    state=previousState(i,:);
    neural=previousNeural(i,:);
    %% Balloon Model
    balloon=@(t,x)balloonModel(t,x,Pa,neural);
    
    %% Solver Differential Equation
    [~,states]=ode45(balloon,timeSpan,state);
    currentState(i,:)=states(size(states,1),:);
    % figure;
    % plot(t,x);
    % legend('s','f','v','q');
end
%% Generate BOLD Signal
V0=0.02;
k1=7*Pa.Rho;
k2=2;
k3=2*Pa.Rho-0.2;
currentBOLD=V0*(k1*(1.-currentState(:,4))+k2*(1.-currentState(:,4)./currentState(:,3))+k3*(1.-currentState(:,3)));

end

function neural=generateNeural(batchSize,probability)
neural=zeros(batchSize,1);
for i=1:batchSize
    if(rand()<probability)
        neural(i,1)=1;
    else
        neural(i,1)=0;
    end
end
end