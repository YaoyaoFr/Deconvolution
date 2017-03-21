function [neural,BOLD]=Simulation(state,TLength,batchSize,Pa)
%% Initial Parameters
if(nargin<=3)
    load('Pa.mat');
end

if(nargin<=2)
    batchSize=1;
end

if(nargin<=1)
    TLength=32;
end

neuralLength=TLength*Pa.Vg+1;

if(nargin<=0)
    state=round(randi(2,batchSize,neuralLength)-1);
end

[batch,neural]=size(state);
if(batch~=batchSize || neural~=neuralLength)
    error('State dimension is error!');
end

load hrf
neural=round(randi(2,batchSize,neuralLength)-1);
BOLD=convn([state(:,2:neuralLength) neural],hrf,'valid');


end
