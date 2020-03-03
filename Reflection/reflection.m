function [flag2, r, theta0] = reflection(l0, d0)

n = [-l0(2) l0(1)]/norm(l0);
k = dot(d0,n);

if (k<0.0001 && k>-0.0001)
    [flag2, r] = deal(0);
    return;
end;


theta0 = acos(abs(dot(d0,l0)/norm(l0)));

r0 = [(d0(1)-2*k*n(1)),(d0(2)-2*k*n(2))];
r = r0/norm(r0);

flag2 = 1;
return;
end