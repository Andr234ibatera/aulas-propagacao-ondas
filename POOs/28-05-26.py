import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time

from typing import Union, Optional, List
from abc import ABC
from enum import Enum


class Medium(ABC):
    def __init__(self, 
                 vp: float, 
                 vs: float, 
                 rho: float, 
                 omega: float, 
                 k_x: float,
        *args: any, **kwargs: any):
        """Class that contains the material medium especification where the wave is propagated.

        Args:
            vp (float): The `speed of compressional waves` in m/s (P-wave), which involve particle motion parallel to the direction of propagation. (Confirm information with the group)
            vs (float): The `speed of shear waves` in m/s (S-wave), which involve particle motion perpendicular to the direction of propagation. (Confirm information with the group)
            rho (float): The `mass per unit volume` in Kg/m**3 of the medium through which the wave travels. (Confirm information with the group)
        """
        super().__init__()
        self.vp: float = vp
        self.vs: float = vs
        self.rho: float = rho
        # Deslocation Field
        self.omega: float = omega
        # Continuit Values
        self.k_x: float = k_x
        self._k_zt: Union[np.float64, np.complex128] = None
        self._k_zl: Union[np.float64, np.complex128] = None
        # Lamé Constants
        self._lame_mu = None
        self._lame_lambda = None
            
    @property
    def lame_mu(self) -> float:
        """The Lamé constants are two independent elastic parameters, `lambda` and `mu`, used to describe the mechanical properties of isotropic materials in wave propagation.
        
        `Mu` is the second Lamé parameter, also called the shear modulus or modulus of rigidity.
        
        Obs: In fluids, where rigidity vanishes `mu = 0`, only lambda governs acoustic wave propagation. In solids, both constants are essential for characterizing stress, strain, and wave behavior. 

        Returns:
            float: returns the second Lamé parameter, also called the shear modulus or modulus of rigidity.
        """
        return self._lame_mu if self._lame_mu else (self.vs**2)*self.rho
    
    @lame_mu.setter
    def lame_mu(self, new_lame_mu) -> None:
        """_summary_

        Args:
            new_lame_mu (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._lame_mu = new_lame_mu
    
    @property
    def lame_lambda(self) -> float:
        """The Lamé constants are two independent elastic parameters, `lambda` and `mu`, used to describe the mechanical properties of isotropic materials in wave propagation.
        
        `Lambda` is known as the first Lamé parameter and represents pure incompressibility.
        
        Obs: In fluids, where rigidity vanishes `mu = 0`, only lambda governs acoustic wave propagation. In solids, both constants are essential for characterizing stress, strain, and wave behavior.

        Returns:
            float: returns the first Lamé parameter and represents pure incompressibility.
        """
        return self._lame_lambda if self._lame_lambda else ((self.vp**2)*self.rho) - (2*self.lame_mu)
    
    @lame_lambda.setter
    def lame_lambda(self, new_lame_lambda) -> None:
        """_summary_

        Args:
            new_lame_lambda (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._lame_lambda = new_lame_lambda
        
    @property
    def k_l(self) -> float:
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self.omega/self.vp
    
    @property
    def k_t(self) -> float:
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self.omega/self.vs
    
    @property
    def k_zt(self) -> Union[np.float64, np.complex128]:
        """_summary_ Get definitions with the group

        Returns:
            Union[np.float64, np.complex128]: _description_
        """
        return self._k_zt if self._k_zt else np.emath.sqrt((self.k_t**2) - (self.k_x**2))
    
    @k_zt.setter
    def k_zt(self, inherited: Union[np.float64, np.complex128]) -> None:
        """_summary_

        Args:
            inherited (Union[np.float64, np.complex128]): _description_
        """
        self._k_zt = inherited
    
    @property
    def k_zl(self) -> Union[np.float64, np.complex128]:
        """_summary_  Get definitions with the group

        Returns:
            Union[np.float64, np.complex128]: _description_
        """
        return self._k_zl if self._k_zl else np.emath.sqrt((self.k_l**2) - (self.k_x**2))
    
    @k_zl.setter
    def k_zl(self, inherited: Union[np.float64, np.complex128]) -> None:
        """_summary_

        Args:
            inherited (Union[np.float64, np.complex128]): _description_
        """
        self._k_zl = inherited
    
    def __str__(self):
        return f"\t\tMaterial Specifications\nVp={self.vp}\tVs={self.vs}\tRho={self.rho}\n\t\tLamé Constants\nLambda={self.lame_lambda}\tMu={self.lame_mu}\n\t\tDisplacement Fields\nOmega={self.omega}\n\t\tKs\nKl={self.k_l}\tKx={self.k_x}\tKt={self.k_t}\tKzt={self.k_zt}\tKzl={self.k_zl}"


class DisplacementPolarizationMatriz(ABC):
    def __init__(self, medium: Medium):
        super().__init__()
        self.medium = medium
    
    @property
    def sin_theta_l(self) -> np.complex128:
        """_summary_

        Returns:
            np.complex128: _description_
        """
        return self.medium.k_x/self.medium.k_l

    @property
    def cos_theta_l(self) -> np.complex128:
        """_summary_

        Returns:
            np.complex128: _description_
        """
        return self.medium.k_zl/self.medium.k_l
    
    @property
    def sin_theta_t(self) -> np.complex128:
        """_summary_

        Returns:
            np.complex128: _description_
        """
        return self.medium.k_x/self.medium.k_t
    
    @property
    def cos_theta_t(self) -> np.complex128:
        """_summary_

        Returns:
            np.complex128: _description_
        """
        return self.medium.k_zt/self.medium.k_t
    
    @property
    def A1(self) -> np.array:
        """Polarization matrix of the displacement going up.

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return np.array([
            [self.sin_theta_l, -self.cos_theta_t, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [self.cos_theta_l, self.sin_theta_t, 0.0 + 0.0j]
        ], dtype=np.complex128)
    
    @property
    def A2(self) -> np.array:
        """Polarization matrix of the displacement going down.

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return self.A1*np.array([[1,1,1], [1,1,1], [-1,-1,1]])
    
    def __str__(self):
        return f"A1={self.A1}\nA2={self.A2}\n\tL\nsin(theta_l)={self.sin_theta_l}\tcos(theta_l)={self.cos_theta_l}\n\tT\nsin(theta_t)={self.sin_theta_t}\tcos(theta_t)={self.cos_theta_t}"


class LMatriz(ABC):
    def __init__(self, medium: Medium, displacement: DisplacementPolarizationMatriz):
        super().__init__()
        self.m = medium
        self.d = displacement
    
    @property
    def L1(self) -> np.array:
        """_summary_

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return np.array([
            [self.m.lame_lambda * (self.m.k_zl*self.d.cos_theta_l + self.m.k_x*self.d.sin_theta_l) + 2*self.m.lame_mu*self.m.k_zl*self.d.cos_theta_l,
            self.m.lame_lambda * (self.m.k_zt*self.d.sin_theta_t - self.m.k_x*self.d.cos_theta_t) + 2*self.m.lame_mu*self.m.k_zt*self.d.sin_theta_t, 
            0.0 + 0.0j],
            [self.m.lame_mu * (self.m.k_x*self.d.cos_theta_l + self.m.k_zl*self.d.sin_theta_l),
            self.m.lame_mu * (self.m.k_x*self.d.sin_theta_t - self.m.k_zt*self.d.cos_theta_t),
            0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, self.m.lame_mu*self.m.k_zt]
        ])
        
    @property
    def L2(self) -> np.array:
        """_summary_

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return self.L1 * np.array([[1,1,1],[-1,-1,1],[1,1,-1]])
    
    def __str__(self) -> str:
        return f"L1\n{self.L1}\nL2\n{self.L2}"


class LocalImpedanceTensor(ABC):
    def __init__(self, medium: Medium, displacement: DisplacementPolarizationMatriz, l_matriz: LMatriz):
        super().__init__()
        self.m = medium
        self.d = displacement
        self.l = l_matriz
    
    @property
    def Z1(self) -> np.array:
        """_summary_

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return -(1/self.m.omega) * self.l.L1 @ np.linalg.inv(self.d.A1)
        
    @property
    def Z2(self) -> np.array:
        """_summary_

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return -(1/self.m.omega) * self.l.L2 @ np.linalg.inv(self.d.A2)
        
    def __str__(self) -> str:
        return f"Z1\n{self.Z1}\nZ2\n{self.Z2}"

class PhiMatriz(ABC):
    def __init__(self, medium: Medium, h: float):
        """_summary_

        Args:
            medium (Medium): _description_
            h (float): _description_
        """
        super().__init__()
        self.m = medium
        self.h = h
    
    @property
    def Phi1(self) -> np.array:
        """_summary_

        Returns:
            np.array: _description_
        """
        return np.array([
            [np.exp(self.m.k_zl*self.h*1.0j), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(self.m.k_zt*self.h*1.0j), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, np.exp(self.m.k_zt*self.h*1.0j)],
        ], dtype=np.complex128)
    
    @property
    def Phi2(self) -> np.array:
        """_summary_

        Returns:
            np.array: _description_
        """
        return np.array([
            [np.exp(-self.m.k_zl*self.h*1.0j), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(-self.m.k_zt*self.h*1.0j), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, np.exp(-self.m.k_zt*self.h*1.0j)],
        ], dtype=np.complex128)
    
    def __str__(self) -> str:
        return f"Ph1\n{self.Phi1}\nPh2\n{self.Phi2}"
    

class MMatriz(ABC):
    def __init__(self, displacement: DisplacementPolarizationMatriz, phi: PhiMatriz):
        super().__init__()
        self.d: DisplacementPolarizationMatriz = displacement
        self.p: PhiMatriz = phi
    
    @property
    def M1(self) -> np.array:
        """_summary_

        Returns:
            np.array: _description_
        """
        return self.d.A1 @ self.p.Phi1 @ np.linalg.inv(self.d.A1)
    
    @property
    def M2(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.d.A2 @ self.p.Phi2 @ np.linalg.inv(self.d.A2)
    
    @property
    def M2Minus(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.d.A2 @ self.p.Phi1 @ np.linalg.inv(self.d.A2)
    
    def __str__(self) -> str:
        return f"M1\n{self.M1}\nM2\n{self.M2}\nM2Minus\n{self.M2Minus}"


def metodo_bissecao(f, a, b, tol=1e-6, max_iter=100):
    """
    Calcula a raiz de f(x) no intervalo [a, b] usando o método da bisseção.
    
    Parâmetros:
    f : callable - Função contínua tal que f(a) * f(b) < 0
    a, b : float - Extremidades do intervalo inicial
    tol : float - Tolerância de erro aceitável
    max_iter : int - Número máximo de iterações
    
    Retorna:
    p : float - Aproximação da raiz
    """
    fa = f(a)
    fb = f(b)
    
    if fa * fb >= 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos.")
    
    for _ in range(max_iter):
        p = (a + b) / 2
        fp = f(p)
        
        # Critério de parada: erro absoluto ou valor da função próximo de zero
        if abs(fp) < tol or (b - a) / 2 < tol:
            return p
        
        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp
            
    return (a + b) / 2

# Exemplo de uso:
# Encontrar a raiz de sin(x) no intervalo [2, 4] (raiz esperada: pi)
raiz = metodo_bissecao(np.sin, 2, 4)
print(f"Raiz encontrada: {raiz}") # Saída próxima de 3.14159...   

def lineplot(lista_x, lista_y, tempo, tolerancia, lblx, lbly) -> None:
    # (Opcional) Define o tamanho da figura
    plt.figure(figsize=(8, 5))

    # 2. Cria o scatter plot
    # color: cor dos pontos | marker: formato do ponto | s: tamanho do ponto
    plt.scatter(lista_x, lista_y, color='blue', marker='o', s=1)

    # 3. Adiciona título e rótulos aos eixos
    plt.title(f"Tempo: {tempo_execucao:.5f} segundos\nTolerância: {tolerancia}")
    plt.xlabel(lblx)
    plt.ylabel(lbly)

    # (Opcional) Adiciona uma grade para facilitar a leitura
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajusta o layout para não cortar nenhuma informação
    plt.tight_layout()

    # 4. Mostra o gráfico na tela
    plt.show()
    
if __name__ == "__main__":
    inicio = time.perf_counter()
    # vp=cl, ct=vs
    medium_data = {
        '3': {'vp': 0.0, 'vs': 0.0, 'rho': 0.0, 'h': np.inf},
        '2': {'vp': 3.0, 'vs': 1.5, 'rho': 7.0, 'h': 1.0},
        '1': {'vp': 0.0, 'vs': 0.0, 'rho': 0.0, 'h': np.inf},
    }
    
    MIN_OMEGA = 1e-3
    MIN_K = 1e-3
    
    MAX_OMEGA = 38.0
    MAX_K = 22.0
    
    DELTA_OMEGA = 1.0
    DELTA_K = 22*1e-2
    
    THRESHOLD = 1
    
    omegas = np.linspace(MIN_OMEGA, MAX_OMEGA, 100)
    ks = np.linspace(MIN_K, MAX_K, 1000)
    
    OMEGA_LBL = f"OMEGA min: {MIN_OMEGA} max: {MAX_OMEGA} len: {len(omegas)}"
    KX_LBL = f"KX min: {MIN_K} max: {MAX_K} len: {len(ks)}"
    
    omega_k_pair = list(it.product(omegas, ks))
    
    # Definindo os meios
    mediums = dict()
    for i in range(3,0,-1):
        medium_temp = list(map(lambda item: 
            Medium(
                vp=medium_data[str(i)]['vp'], 
                vs=medium_data[str(i)]['vs'], 
                rho=medium_data[str(i)]['rho'],
                omega=item[0],
                k_x=item[1]
            ), omega_k_pair))
        mediums.update({str(i): medium_temp})
        
    # Definindo as matrizes de deslocamento
    displacements = dict()
    for i in range(3,0,-1):
        displacements.update({str(i):
            list(map(lambda m: DisplacementPolarizationMatriz(medium=m), mediums[str(i)]))})
    
    # L Matriz from the top to bottom
    ls = dict()
    for i in range(3,0,-1):
        ls.update({str(i):
            list(map(lambda m, d: LMatriz(medium=m, displacement=d), mediums[str(i)], displacements[str(i)]))})
    
    # Local Impedance Tensor from the top to bottom
    zs = dict()
    for i in range(3,0,-1):
        zs.update({str(i):
            list(map(lambda m, d, l: LocalImpedanceTensor(medium=m, displacement=d, l_matriz=l), mediums[str(i)], displacements[str(i)], ls[str(i)]))})
    
    # Defining Phis for the layer 2 to 3
    phis = dict()
    for i in range(2,3):
        phis.update({str(i): list(map(lambda m: PhiMatriz(medium=m, h=medium_data[str(i)]['h']), mediums[str(i)]))})
    
    # Defining MMatraiz for layers 2 to 3
    ms = dict()
    for i in range(2,3):
        ms.update({str(i): list(map(lambda d, p: MMatriz(displacement=d, phi=p), displacements[str(i)], phis[str(i)]))})
        
    # Defining Gs, Rs, Ws
    gs = dict()
    rs = dict()
    gs.update({'1': list(map(lambda _: np.array([[x]*3 for x in [0.0 + 0.0j]*3], dtype=np.complex128), zs['1']))})
    rs.update({'1': list(map(lambda z, g: np.linalg.inv((z.Z1 - g))@(g - z.Z2), zs['2'], gs['1']))})
    ws = dict()
    for i in range(2,3):
        w_temp = list(map(lambda m, r: m.M1 @ r @ m.M2Minus, ms[str(i)], rs[str(i-1)]))
        ws.update({str(i): w_temp})
        g_temp = list(map(lambda z, w: (z.Z1@w+z.Z2) @ np.linalg.inv(w+np.identity(z.Z2.shape[0])), zs[str(i)], w_temp))
        gs.update({str(i): g_temp})
    
    # Defining G2 determinants
    dets = list(map(lambda g: np.linalg.det(g), gs['2']))
    
    # list(map(lambda m, d: m.__setatr__("determinant", d), mediums['2'], dets))
    
    # index, value_1[sem o último], value_2[sem o primeiro]
    indiced = list(map(lambda item: (item[2][0], item[0], item[1]), zip(dets[:-1], dets[1:], enumerate(dets[:-1]))))
    # identificando se valores consecutivos possuem sinais diferentes
    # index, (valor_1, valor_2), if sinal diferente
    signed = list(map(lambda item: (
        item[0], 
        (item[1], item[2]), 
        (item[1]>=0 and item[2]<0) or (item[1]<0 and item[2]>=0)
    ), indiced))
    # filtragem para pegar sinais diferentes
    filtered_signed = list(filter(lambda item: item[2], signed))
    # correlacionando com a diferença entre determinantes
    # index, (valor_1, valor_2), sinal, diferença
    diffs = list(map(lambda item: (
        item[0], item[1], item[2],
        abs(np.abs(item[1][0]) - np.abs(item[1][1])),
    ), filtered_signed))
    # pegando apenas valores da diferênça abaixo do treshold
    filtered_diffs = list(filter(lambda item: item[3]<=THRESHOLD, diffs))
    
    # selecting omegas and kxs
    kx_omega_pair = list(map(lambda item: 
        (
            (mediums['2'][item[0]].k_x + mediums['2'][item[0]+1].k_x)/2,
            mediums['2'][item[0]].omega
        ), 
    filtered_diffs))
    
    fim = time.perf_counter()
    tempo_execucao = fim - inicio
    
    kxs = list(map(lambda item: item[0], kx_omega_pair))
    omegas = list(map(lambda item: item[1], kx_omega_pair))
    
    lineplot(
        lista_x=kxs, 
        lista_y=omegas, 
        tempo=tempo_execucao, 
        tolerancia=THRESHOLD,
        lblx=KX_LBL, 
        lbly=OMEGA_LBL)
    
    pass    