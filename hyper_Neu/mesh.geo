//Définition des valeurs physiques
L = 0.7654; //Longeur selon x
H = 4.443; //Longuer selon y
h = 0.1; //0.05; //0.1; //Taille du maillage
aux = h/5;

Point(1) = {0,0,0,h};
Point(2) = {aux,0,0,h};
Point(3) = {0,aux,0,h};
Point(4) = {0,H,0,h};
Point(5) = {0,H-aux,0,h};
Point(6) = {aux,H,0,h};
Point(7) = {L,0,0,h};
Point(8) = {L-aux,0,0,h};
Point(9) = {L,aux,0,h};
Point(10) = {L,H,0,h};
Point(11) = {L-aux,H,0,h};
Point(12) = {L,H-aux,0,h};

Line(1) = {1,2};
Line(2) = {2,8};
Line(3) = {8,7};
Line(4) = {7,9};
Line(5) = {9,12};
Line(6) = {12,10};
Line(7) = {10,11};
Line(8) = {11,6};
Line(9) = {6,4};
Line(10) = {4,5};
Line(11) = {5,3};
Line(12) = {3,1};

Line Loop(1) = {1:12};

Plane Surface(1) = {1};

Physical Line(1) = {1,12};
Physical Line(2) = {3,4};
Physical Line(3) = {6,7};
Physical Line(4) = {9,10};

Physical Surface(1) = {1};