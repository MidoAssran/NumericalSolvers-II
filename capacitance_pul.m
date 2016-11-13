clear all;

potentials = SIMPLE2D_M('node.dat', 'tri.dat', 'bc.dat')

S = [1.0000   -0.5000         0   -0.5000
    -0.5000    1.0000   -0.5000         0
          0   -0.5000    1.0000   -0.5000
    -0.5000         0   -0.5000    1.0000];

permittivity = 8.85418782e-12;
const = 0.5 * permittivity;

boundary = [4,10,16,22,28,34];
W = 0;

for i = 1:28
    
    if any(i == boundary)
        continue
    end
    
    v = [i, i+1, i+7, i+6];
    all_U = potentials(:,4);
    U = [all_U(v(1)); all_U(v(2)); all_U(v(3)); all_U(v(4))];
    
    W = W + (U' * S * U);
end

W = W .* const;

C = 4* W / (0.5 * (110^2))