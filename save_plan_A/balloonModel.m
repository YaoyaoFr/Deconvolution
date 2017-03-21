function dx_dt=balloonModel(t,x,Pa,e)

%% Get Parameters

%Biophysical Parameters
Kappa=Pa.Kappa;
Gamma=Pa.Gamma;
Tau=Pa.Tau;
Alpha=Pa.Alpha;
Rho=Pa.Rho;
Vg=Pa.Vg;

% Hemodynamic State
s=x(1);
f=x(2);
v=x(3);
q=x(4);



%% Differential Equations
ds_dt=e-Kappa*s-Gamma*(f-1);
df_dt=s;
dv_dt=1/Tau*(f-v^(1/Alpha));
dq_dt=1/Tau*Efun(f,Rho)/Rho-v^(1/Alpha-1)*q;

dx_dt=[ds_dt;df_dt;dv_dt;dq_dt];


end

%% Oxygen Extraction
function E=Efun(f,Rho)
E=1-(1-Rho)^(1/f);
end


%% Initation The Nueral Coding
function z=Zfun(t,Vg,e)
if(t<0.5/Vg)
    z=e;
else
    z=0;
end
end