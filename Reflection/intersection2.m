function [flag, t] = intersection2(o, d, a, b)

l = a-b;
norm_l = [-l(2) l(1)];

parallel_check = dot(norm_l, d);

if (parallel_check<=0.0001 && parallel_check>=-0.0001)
    [flag, t] = deal(0);
    return;
end;


y0 = abs(d(1));

if (y0<0.0001)
    
    k = (o(1)-a(1))/(b(1)-a(1));
    
    
else
    nomi = (a(2)-o(2))*d(1) - (a(1)-o(1))*d(2);
    deno = (b(1)-a(1))*d(2) - (b(2)-a(2))*d(1);
    k = nomi/deno;
end;



if (y0<0.0001)
    
    t = (a(2)-o(2)+k*(b(2)-a(2)))/d(2);
    
else if (d(2) ~= 0 && d(1) ~= 0)
        t = ((a(1)-o(1)) + k*(b(1)-a(1)))/d(1);
    else if (d(1) == 0)
            t = ((a(2)-o(2)) + k*(b(2)-a(2)))/d(2);
        else
            t = ((a(1)-o(1)) + k*(b(1)-a(1)))/d(1);
        end;
    end;
end;


if (t<0.001)
    [flag, t] = deal(0);
    return;
else if (k<-0.01 || k>1.01)
        [flag, t] = deal(0);
        return;
    else if (k>0.01 && k<0.99)
            flag = 1;
        else
            flag = -1;
        end;
    end;
end;


return;
end




