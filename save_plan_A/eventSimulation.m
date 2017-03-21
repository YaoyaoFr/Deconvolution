function [ output_args ] = eventSimulation(timeLength,continued, interval, Pa )
%EVENTSIMULATION 此处显示有关此函数的摘要
%   此处显示详细说明
if(~exist('Pa','var'))
    load Pa;
end

if(~exist('interval','var'))
    interval=10;
end

if(~exist('continued','var'))
    continued=2;
end

if(~exist('timeLength','var'))
    timeLength=12;
end

epoch=floor(timeLength/(interval+continued));
neural=[];
for i=1:epoch
    neural=[neural ones(1,continued*Pa.Vg) zeros(1,interval*Pa.Vg)];
end
currentState=[0,1,1,1];
state=[];
BOLD=[];
for i=1:length(neural)
    [previousNeural,previousState,currentState,currentBOLD]=nextState(currentState,neural(i),1);
    state=[state;currentState];
    BOLD(i)=currentBOLD;
end

v=state(:,3);
q=state(:,4);
BOLD=BOLD';

plot3(v,q,BOLD)

hold on

V0=0.02;
k1=7*Pa.Rho;
k2=2;
k3=2*Pa.Rho-0.2;
syms x y
z=V0*(k1*(1-y)+k2*(1-y/x)+k3*(1-y));
ezsurf(z,[0,2])
end

