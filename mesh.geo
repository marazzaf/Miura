//Définition des valeurs physiques
theta = pi/2;
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)));
alpha = sqrt(1 / (1 - sin(theta/2)**2));
l = 2*pi/alpha;
L = 1e-3; //Longeur selon x
H = 1e-3; //Longuer selon y
h = 1.1e-5; //Taille du maillage

Point(1) = {0,H/2,0,h};
Point(2) = {0,-H/2,0,h};
Point(3) = {L,-H/2,0,h};
Point(4) = {L,H/2,0,h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

Physical Line(4) = {4}; //Homogeneous Dirichlet boundary
Physical Line(53) = {5,3}; //Homogeneous Neumann
Physical Line(1) = {1}; //First nonhomogeneous Dirichlet
Physical Line(2) = {2}; //Second nonhomogeneous Dirichlet

Physical Surface(11) = {1};