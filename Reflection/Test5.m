s7 = [20 45]; s8 = [35 35]; l1 = s7 - s8;

s3 = [35 35]; s4 = [35 20]; l2 = s3 - s4;

s5 = [15 10]; s6 = [50 10]; l3 = s5 - s6;

s1 = [50 20]; s2 = [15 60]; l4 = s1 - s2;


S = [s1; s2; s3; s4; s5; s6; s7; s8];
% s1, s2 are the co-ordinates of the first wall.
% s3, s4 the next wall. s7 s8 the next wall. s5 s6 another wall
L = [l1; l2; l3; l4];
m = length(L);


% delta_angl = pi/180;
% theta = 0:delta_angl:2*pi;
% Tx = [cos(theta); sin(theta)];
% Z = length(Tx);
% z = 270;     %Transmitting direction in degree
% direction = [Tx(1,z+1), Tx(2,z+1)];

origin = [36 19];   %Tx position

input_direction = [1 2];
direction = input_direction/norm(input_direction); 
% Unit vector in direction of original ray

reflect_lmt = 10;   
% reflection level up-limits ( Maximum number of reflections.


Distances = zeros(reflect_lmt+1, 1);
diffract_end_distance = zeros(reflect_lmt, 1);

Origins = zeros(reflect_lmt+1, 2);
Origins(1,1) = origin(1); Origins(1,2) = origin(2);

Directions = zeros(reflect_lmt+1, 2);
Directions(1,1) = direction(1); Directions(1,2) = direction(2);



for u = 1:reflect_lmt
    
    
    T = zeros(1,m);
    Diff = zeros(1,m);
    
    
    for l=1:m
        
        sa = S(2*l-1,:);
        sb = S(2*l,:);
        % sa-sb corresponds to a wall.
        
        [flag, t] = intersection2(origin, direction, sa, sb);
        
        
        if (flag==1)
            T(l) = t;
        end;
        
        if (flag==-1)
            Diff(l) = t;
        end;
        
        
    end;
    
    diffract_indi = any(Diff);
    reflect_indi = any(T);
    
    T(T==0) = 9999;
    
    
    if (diffract_indi)
        
        Diff(Diff==0) = 9999;
        
        Diff0 = min(Diff);
        T0 = min(T);
        
        t0 = min(Diff0,T0);
        
        if (t0==Diff0)
            true_diffract = 1;
            t = Diff0;
        else
            true_diffract = 0;
            [t, i0] = min(T);
            l = L(i0,:);
        end;
        
    else
        true_diffract = 0;
        
        [t, i0] = min(T);
        l = L(i0,:);
    end;
    
    
    if (true_diffract==1)
        
        flag2 = 0;
        r = zeros(1,2);
        Distances(u) = 0;
        diffract_end_distance(u) = t;
        t = 0;
        
    else if (reflect_indi)
            
            [flag2, r] = reflection(l,direction);
            
            Distances(u) = t;
            diffract_end_distance(u) = 0;
            
        else
            [flag, t, Distances(u), flag2, diffract_end_distance(u)] = deal(0);
            r = direction;
        end;
        
    end;
    
    
    p = origin + t*direction;
    
    
    origin = p;
    Origins(u+1,1) = p(1); Origins(u+1,2) = p(2);
    
    direction = r;
    Directions(u+1,1) = r(1); Directions(u+1,2) = r(2);
    
end;






figure;
hold on;
grid on;

text(Origins(1,1),Origins(1,2),'origin');
plot(Origins(1,1),Origins(1,2),'r.','MarkerSize',15);

for j=1:m
    
    a = min(S(2*j-1,1),S(2*j,1));
    b = max(S(2*j-1,1),S(2*j,1));
    
    if (a~=b)
        x = a:0.01:b;
        y = ((S(2*j,2)-S(2*j-1,2))/(S(2*j,1)-S(2*j-1,1)))*(x-S(2*j-1,1))+S(2*j-1,2);
        plot(x,y,'k-','linewidth',2);
    else
        a0 = min(S(2*j-1,2),S(2*j,2));
        b0 = max(S(2*j-1,2),S(2*j,2));
        y = a0:0.01:b0;
        plot(a*ones(size(y)),y,'k-','linewidth',2);
    end;
    
end;


diff_end_indi = any(diffract_end_distance);

if (diff_end_indi)
    
    v0 = find(diffract_end_distance);
    V = v0 - 1;
    
else
    V = reflect_lmt;
end;


for v = 1:V
    
    if (Distances(v)~=0)
        
        quiver(Origins(v,1),Origins(v,2),Directions(v,1),Directions(v,2),Distances(v),'.');
        
        plot(Origins(v+1,1),Origins(v+1,2),'g.','MarkerSize',15);
        
    end;
    
end;


if (diff_end_indi)
    
    diffract_end_distance0 = diffract_end_distance(diffract_end_distance~=0);
    quiver(Origins(v0,1),Origins(v0,2),Directions(v0,1),Directions(v0,2),diffract_end_distance0,'color','green');
    
else
    quiver(Origins(reflect_lmt+1,1),Origins(reflect_lmt+1,2),Directions(reflect_lmt+1,1),Directions(reflect_lmt+1,2),15,'color','red');
end;



%alpha(0.5);
axis equal;
xlabel('x');
ylabel('y');
