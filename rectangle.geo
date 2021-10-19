//Définition des valeurs physiques
L = 0.76536; //Longeur selon x
H = 1.4142; //Longuer selon y
h = 0.1; //Taille du maillage

Point(1) = {-L/2,H,0,h};
Point(2) = {-L/2,0,0,h};
Point(3) = {L/2,0,0,h};
Point(4) = {L/2,H,0,h};

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