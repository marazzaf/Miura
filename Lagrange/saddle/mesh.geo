L = 2;
W = 2;
h = 0.0125;

Point(1) = {-L/2, -W/2, 0, h};
Point(2) = {-L/2, W/2, 0, h};
Point(4) = {L/2, -W/2, 0, h};
Point(3) = {L/2, W/2, 0, h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(5) = {1:4};

Plane Surface(1) = {5};

Physical Surface(1) = {1};

Physical Curve(2) = {2, 1, 4, 3};
