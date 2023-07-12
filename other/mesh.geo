L = 10;
W = 2;
h = W/25; //0.025; //0.0125;

Point(1) = {-L/2, -W/2, 0, h};
Point(2) = {-L/2, W/2, 0, h};
Point(4) = {L/2, -W/2, 0, h};
Point(3) = {L/2, W/2, 0, h};
Point(5) = {L/2, -W/6, 0, h};
Point(6) = {-L/2, W/6, 0, h};
Point(7) = {L/2, W/6, 0, h};
Point(8) = {-L/2, -W/6, 0, h};

Line(1) = {1,8};
Line(2) = {8,6};
Line(3) = {6,2};
Line(4) = {2,3};
Line(5) = {3,7};
Line(6) = {7,5};
Line(7) = {5,4};
Line(8) = {4,1};

Line Loop(9) = {1:8};

Plane Surface(1) = {9};

Physical Surface(1) = {1};

Physical Curve(1) = {2};
Physical Curve(2) = {6};
