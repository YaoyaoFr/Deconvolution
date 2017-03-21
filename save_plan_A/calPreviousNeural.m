function [previousNeural] = calPreviousNeural( previousState,currentState,Pa)
if(nargout<3)
    load Pa;
end

Vg=Pa.Vg;
Kappa=Pa.Kappa;
Gamma=Pa.Gamma;

previousNeural=(currentState(:,1)-previousState(:,1))*Vg+Kappa*previousState(:,1)+Gamma*(previousState(:,2)-1);

end

