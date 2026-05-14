import numpy as np


# TODO calcular o coeficiente de reflexão em dois semi espaços
# G tensor de impedância das ondas indo para baixo

# 1 - especificar os materiais
# Meio 2 em cima
vp_2 = 2000 # m/s
vs_2 = 1500 # m/s
rho_2 = 5000 # Kg/m**3

# Meio 1 em baixo
vp_1 = 1000 # m/s
vs_1 = 700 # m/s
rho_1 = 2500 # Kg/m**3

# Constantes de Lamé
mu_1 = (vs_1**2)*rho_1
# mu_2 = (vs_2**2)*rho_2
mu_2 = 10.0e3 # Pa

Lambda_1 = ((vp_1**2)*rho_1) - (2*mu_1)
# Lambda_2 = ((vp_2**2)*rho_2) - (2*mu_2)
Lambda_2 = mu_2  # Pa

# Campo de deslocamento
theta_i = (30*np.pi)/180 # angulo de incidência 30 graus
omega = (2*np.pi)*100 # frequência de 100 Hertz da onda P
# Meio superior
k2_l = omega/vp_2 # Vetor de onda incidente
k_x = k2_l*np.sin(theta_i)
k2_t = omega/vs_2
k2_zt = np.emath.sqrt((k2_t**2) - (k_x**2)) # pode ser complexo
k2_zl = np.emath.sqrt((k2_l**2) - (k_x**2)) # pode ser complexo
# A2_1 (Matriz A do meio superior associado a onda indo para cima)
# sin(theta_l) = k_x/k2_l [isso é A1_11]
sin_theta_l2 = k_x/k2_l
# -cos(theta_t) = -(k2_zt/k2_t) [isso é A1_12]
cos_theta_t2 = k2_zt/k2_t
# 0 [isso é A1_13, A1_21, A1_22, A1_33]
# 1 [isso é A1_23]
# cos(theta_l) = k2_zl/k2_l [isso é A1_31]
cos_theta_l2 = k2_zl/k2_l
# sin(theta_t) = k_x/k2_t [isso é A1_32]
sin_theta_t2 = k_x/k2_t
# Quando montar a matriz A precisa considerar que é complexo
A2_1 = np.array([
    [sin_theta_l2,      -cos_theta_t2,      0.0 + 0.0j],
    [0.0 + 0.0j,    0.0 + 0.0j,         1.0 + 0.0j],
    [cos_theta_l2,    sin_theta_t2,           0.0 + 0.0j]
], dtype=np.complex128)

# A2_2 (Matriz A do meio superior associado a onda indo para baixo)
# sin(theta_l) = k_x/k2_l [isso é A2_11]
# -cos(theta_t) = -(k2_zt/k2_t) [isso é A2_12]
# 0 [isso é A2_13, A2_21, A2_22, A2_33]
# 1 [isso é A2_23]
# -cos(theta_l) = -(k2_zl/k2_l) [isso é A2_31]
# -sin(theta_t) = -(k_x/k2_t) [isso é A2_32]
A2_2 = A2_1 * np.array([[1,1,1],[1,1,1],[-1,-1,1]])


# Meio inferior
k1_l = omega/vp_1 # Vetor de onda transmitida
k1_t = omega/vs_1
k1_zt = np.emath.sqrt((k1_t**2) - (k_x**2)) # pode ser complexo
k1_zl = np.emath.sqrt((k1_l**2) - (k_x**2)) # pode ser complexo

# A1_1 (Matriz A do meio inferior associado a onda indo para cima)
# sin(theta_l) = k1_x/k1_l [isso é A1_11]
sin_theta_l1 = k_x/k1_l
# -cos(theta_t) = -(k1_zt/k1_t) [isso é A1_12]
cos_theta_t1 = k1_zt/k1_t
# 0 [isso é A1_13, A1_21, A1_22, A1_33]
# 1 [isso é A1_23]
# cos(theta_l) = k1_zl/k1_l [isso é A1_31]
cos_theta_l1 = k1_zl/k1_l
# sin(theta_t) = k_x/k1_t [isso é A1_32]
sin_theta_t1 = k_x/k1_t
# Quando montar a matriz A precisa considerar que é complexo
A1_1 = np.array([
    [sin_theta_l1,     -cos_theta_t1,   0.0 + 0.0j],
    [0.0 + 0.0j,    0.0 + 0.0j,     1.0 + 0.0j],
    [cos_theta_l1,    sin_theta_t1,       0.0 + 0.0j]
], dtype=np.complex128)

# A1_2 (Matriz A do meio inferior associado a onda indo para baixo)
# sin(theta_l) = k1_x/k1_l [isso é A2_11]
# -cos(theta_t) = -(k1_zt/k1_t) [isso é A2_12]
# 0 [isso é A2_13, A2_21, A2_22, A2_33]
# 1 [isso é A2_23]
# -cos(theta_l) = -(k1_zl/k1_l) [isso é A2_31]
# -sin(theta_t) = -(k_x/k1_t) [isso é A2_32]
A1_2 = A1_1 * np.array([[1,1,1],[1,1,1],[-1,-1,1]])


C_p = 1.0 #(amplitude onda C_p=1)
u_down =  np.array([[C_p*np.sin(theta_i)], [0.0], [-C_p*np.cos(theta_i)]], dtype=np.float64) # Vetor campo de deslocamento descendo (amplitude onda C_p=1)

# Matriz que relaciona as amplitudes ao traction vector do meio 1
L1_1 = np.array([
    [Lambda_1 * (k1_zl*cos_theta_l1+k_x*sin_theta_l1) + 2*mu_1*k1_zl*cos_theta_l1,
     Lambda_1 * (k1_zt*sin_theta_t1 - k_x*cos_theta_t1) + 2*mu_1*k1_zt*sin_theta_t1,
     0.0 + 0.0j],
    [mu_1*(k_x*cos_theta_l1 + k1_zl*sin_theta_l1),
     mu_1*(k_x*sin_theta_t1 - k1_zt*cos_theta_t1),
     0.0 + 0.0j],
    [0.0 + 0.0j, 0.0 + 0.0j, mu_1*k1_zt]
], dtype=np.complex128)

L1_2 = L1_1 * np.array([[1,1,1],[-1,-1,1],[1,1,-1]])

# Matriz que relaciona as amplitudes ao traction vector do meio 2
L2_1 = np.array([
    [Lambda_2 * (k2_zl*cos_theta_l2+k_x*sin_theta_l2) + 2*mu_2*k2_zl*cos_theta_l2,
     Lambda_2 * (k2_zt*sin_theta_t2 - k_x*cos_theta_t2) + 2*mu_2*k2_zt*sin_theta_t2,
     0.0 + 0.0j],
    [mu_2*(k_x*cos_theta_l2 + k2_zl*sin_theta_l2),
     mu_2*(k_x*sin_theta_t2 - k2_zt*cos_theta_t2),
     0.0 + 0.0j],
    [0.0 + 0.0j, 0.0 + 0.0j, mu_2*k2_zt]
], dtype=np.complex128)

L2_2 = L2_1 * np.array([[1,1,1],[-1,-1,1],[1,1,-1]])

# Tensor de Impedância local
Z1_1 = -(1/omega)*L1_1@np.linalg.inv(A1_1)
Z1_2 = -(1/omega)*L1_2@np.linalg.inv(A1_2)
Z2_1 = -(1/omega)*L2_1@np.linalg.inv(A2_1)
Z2_2 = -(1/omega)*L2_2@np.linalg.inv(A2_2)

# Tensor de impedância superficial
# G = Z1_2
G = np.array([
    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
])

# Tensor de reflexão
R = np.linalg.inv((Z2_1 - G))@(G - Z2_2)

if __name__ == "__main__":
    pass

# matheus   matheus
# adriyel   AdriyelRGFM 
# Guilherme GuilhermePiorno