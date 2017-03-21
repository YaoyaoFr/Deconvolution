function [ BOLD ] = calBOLD( state )
load Pa

V0=0.02;
k1=7*Pa.Rho;
k2=2;
k3=2*Pa.Rho-0.2;
BOLD=V0*(k1*(1.-state(:,4))+k2*(1.-state(:,4)./state(:,3))+k3*(1.-state(:,3)));


end

