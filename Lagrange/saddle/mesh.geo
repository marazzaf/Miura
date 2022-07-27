//Définition des valeurs physiques
L = 2; //Longeur selon x
H = 1; //Longuer selon y
h = 0.05; //0.05; //0.01; //Taille du maillage

Point(1) = {0,0,0,h};
Point(2) = {0,H,0,h};
Point(3) = {L,H,0,h};
Point(4) = {L,0,0,h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};

Physical Surface(1) = {1};