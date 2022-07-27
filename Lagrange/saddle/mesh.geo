//Définition des valeurs physiques
h = 0.025; //0.05; //0.025; //Taille du maillage

Point(1) = {0,0,0,h};
Point(2) = {0,1,0,h};
Point(3) = {-1,0,0,h};
Point(4) = {0,-1,0,h};
Point(5) = {1,0,0,h};

Circle(1) = {3,1,4};
Circle(2) = {4,1,5};
Circle(3) = {5,1,2};
Circle(4) = {2,1,3};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

Physical Line(1) = {1:4};

Physical Surface(1) = {1};