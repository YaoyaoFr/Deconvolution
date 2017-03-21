function [ Jacobian_g ] = jacobian_g( state )
load Pa;


V0=0.02;
k1=7*Pa.Rho;
k2=2;
k3=2*Pa.Rho-0.2;

v=state(:,3);
q=state(:,4);

Jacobian_g=[V0*(k2*q./v.^2.-k3) V0*(-k1-k2./v)];


end

