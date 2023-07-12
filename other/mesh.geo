L = 2;
W = 2;
h = 0.025; //0.0125;

Point(1) = {-L/2, -W/2, 0, h};
Point(2) = {-L/2, W/2, 0, h};
Point(4) = {L/2, -W/2, 0, h};
Point(3) = {L/2, W/2, 0, h};
Point(5) = {L/10, -W/2, 0, h};
Point(6) = {-L/10, W/2, 0, h};
Point(7) = {L/10, W/2, 0, h};
Point(8) = {-L/10, -W/2, 0, h};

Line(1) = {1,2};
Line(2) = {2,6};
Line(3) = {6,7};
Line(4) = {7,3};
Line(5) = {3,4};
Line(6) = {4,5};
Line(7) = {5,8};
Line(8) = {8,1};

Line Loop(9) = {1:8};

Plane Surface(1) = {9};

Physical Surface(1) = {1};

Physical Curve(1) = {2};
Physical Curve(2) = {6};
