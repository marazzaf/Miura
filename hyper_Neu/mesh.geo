//Définition des valeurs physiques
L = 0.7654; //Longeur selon x
H = 4.443; //Longuer selon y
h = 0.1; //0.05; //0.1; //Taille du maillage
aux = h/5;

Point(1) = {0,0,0,h};
Point(2) = {aux,0,0,h};
Point(3) = {0,aux,0,h};
Point(4) = {0,H,0,h};
Point(5) = {L,0,0,h};
Point(6) = {L,H,0,h};

Line(1) = {1,2};
Line(2) = {2,5};
Line(3) = {5,6};
Line(4) = {6,4};
Line(5) = {4,3};
Line(6) = {3,1};

Line Loop(1) = {1:6};

Plane Surface(1) = {1};

Physical Line(1) = {1,6};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};

Physical Surface(1) = {1};