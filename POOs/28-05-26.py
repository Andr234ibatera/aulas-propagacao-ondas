import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time

from typing import Union, Optional, List, Dict, Tuple
from abc import ABC
from enum import Enum
from tqdm import tqdm


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


def calculate_until_det_of_G(
    vp: float, 
    vs: float, 
    rho: float, 
    h: float, 
    omega: float, 
    kx: float,
    G1: np.ndarray
) -> Dict:
    # Definindo os meios
    medium = Medium(vp=vp, vs=vs, rho=rho, omega=omega, k_x=kx)
    # Definindo as matrizes de deslocamento
    displacement = DisplacementPolarizationMatriz(medium=medium) 
    # L Matriz from the top to bottom
    l_matriz = LMatriz(medium=medium, displacement=displacement)
    # Local Impedance Tensor from the top to bottom
    local_impedance = LocalImpedanceTensor(medium=medium, displacement=displacement, l_matriz=l_matriz)
    # Defining Phis for the layer 2 to 3
    phi_matriz = PhiMatriz(medium=medium, h=h)
    # Defining MMatraiz for layers 2 to 3
    m_matriz = MMatriz(displacement=displacement, phi=phi_matriz)  
    # Defining Gs, Rs, Ws
    r1 = np.linalg.inv((local_impedance.Z1 - G1))@(G1 - local_impedance.Z2)
    w = m_matriz.M1 @ r1 @ m_matriz.M2Minus
    g = (local_impedance.Z1@w + local_impedance.Z2) @ np.linalg.inv(w+np.identity(local_impedance.Z2.shape[0]))
    # Calculating the determinat of g
    det = np.linalg.det(g)
    
    return {
        "vp": vp, "vs": vs, "rho": rho, "h": h, "omega": omega, "kx": kx,
        "medium": medium,
        "displacement": displacement,
        "l_matriz": l_matriz,
        "local_impedance": local_impedance,
        "phi_matriz": phi_matriz,
        "m_matriz": m_matriz,
        "r1": r1,
        "w": w,
        "g": g,
        "det": det
    }


def metodo_bissecao(d_1: Dict, d_2: Dict, G1: np.ndarray, tol=1e-6, max_iter=100) -> Tuple[float, float]:
    """_summary_

    Args:
        d_1 (Dict): _description_
        d_2 (Dict): _description_
        G1 (np.ndarray): _description_
        tol (_type_, optional): _description_. Defaults to 1e-6.
        max_iter (int, optional): _description_. Defaults to 100.

    Returns:
        Tuple[float, float]: _description_
    """
    kx_1 = d_1['medium'].k_x
    kx_2 = d_2['medium'].k_x
    
    for i in range(max_iter):
        
        kx_mid = (kx_1+kx_2)/2
        
        d_mid = calculate_until_det_of_G(
                vp=d_1['medium'].vp,
                vs=d_1['medium'].vs,
                rho=d_1['medium'].rho,
                h=d_1['phi_matriz'].h,
                omega=d_1['medium'].omega,
                kx=kx_mid,
                G1=G1,
            )
        
        if abs(d_mid['det'])<tol or ((kx_2-kx_1)/2)<tol:
            break
        
        if (d_1['det']*d_mid['det'])<0:
            d_2 = d_mid
        else:
            d_1 = d_mid
    
    return d_mid['medium'].k_x, d_mid['medium'].omega


def plotting(dados: Dict, x_lbl: str, y_lbl: str, tol: float, time: float):
    # Iterando sobre o dicionário (chave = grupo, valor = lista de tuplas)
    for grupo, pontos in dados.items():
        # Separando as coordenadas X e Y usando list comprehension
        x = [p[0] for p in pontos]
        y = [p[1] for p in pontos]
        
        # Plota os pontos do grupo atual. 
        # O matplotlib define uma cor automática diferente para cada iteração.
        plt.scatter(x, y, label=grupo, alpha=0.8, edgecolors='none', s=20)

    # Customizações do gráfico
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(f'Tolerance before bisection: {tol}\nExecution Time: {time} sec')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Exibe a legenda indicando qual cor pertence a qual grupo
    # plt.legend(title="Omegas")

    # Exibe o gráfico na tela
    # plt.show()

    # # Ajusta o layout para não cortar nenhuma informação
    plt.tight_layout()

    # 4. Mostra o gráfico na tela
    plt.savefig(f'tolerancia{tol}.png', dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    # vp=cl, ct=vs
    medium_data = {
        '3': {'vp': 0.0, 'vs': 0.0, 'rho': 0.0, 'h': np.inf},
        '2': {'vp': 3.0, 'vs': 1.5, 'rho': 7.0, 'h': 1.0},
        '1': {'vp': 0.0, 'vs': 0.0, 'rho': 0.0, 'h': np.inf},
    }
    
    DEBUG = False
    THRESHOLD = 1
    
    if DEBUG:
        MIN_OMEGA = 1e-3
        MAX_OMEGA = 38.0
        DELTA_OMEGA = 1.0
        NUM_OMEGAS = int((MAX_OMEGA - MIN_OMEGA) // DELTA_OMEGA) + 1
        REAL_LAST_OMEGA = MIN_OMEGA + (NUM_OMEGAS - 1) * DELTA_OMEGA
        
        MIN_K = 1e-3
        MAX_K =  22
        DELTA_K = 22*1e-2
        NUM_KS = int((MAX_K - MIN_K) // DELTA_K) + 1
        REAL_LAST_K = MIN_K + (NUM_KS - 1) * DELTA_K
    else:
        # omega do código do professor com guilherme
        # 1e-4:2.5e-2:14
        MIN_OMEGA = 1e-4
        MAX_OMEGA = 14.0
        DELTA_OMEGA = 2.5e-2
        NUM_OMEGAS = int((MAX_OMEGA - MIN_OMEGA) // DELTA_OMEGA) + 1
        REAL_LAST_OMEGA = MIN_OMEGA + (NUM_OMEGAS - 1) * DELTA_OMEGA
        # kx do código do professor com guilherme
        # 0.000001:0.0000001:0.001
        MIN_K = 1e-6
        MAX_K = 1e-3
        # DELTA_K = 1e-7
        DELTA_K = 1e-5
        NUM_KS = int((MAX_K - MIN_K) // DELTA_K) + 1
        REAL_LAST_K = MIN_K + (NUM_KS - 1) * DELTA_K

    G1 = np.zeros((3, 3), dtype=np.complex128)

    inicio = time.perf_counter()
        
    omegas = np.linspace(MIN_OMEGA, REAL_LAST_OMEGA, NUM_OMEGAS)
    ks = np.linspace(MIN_K, REAL_LAST_K, NUM_KS)
        
    OMEGA_LBL = f"OMEGA min: {MIN_OMEGA} max: {MAX_OMEGA} len: {len(omegas)}"
    KX_LBL = f"KX min: {MIN_K} max: {MAX_K} len: {len(ks)}"
    
    omega_k_pair = list(it.product(omegas, ks))
    
    calculated_data = list(map(lambda item: 
        calculate_until_det_of_G(
            vp=medium_data['2']['vp'],
            vs=medium_data['2']['vs'],
            rho=medium_data['2']['rho'],
            h=medium_data['2']['h'],
            omega=item[0],
            kx=item[1],
            G1=G1,
        )
    , tqdm(omega_k_pair, desc="Generating initial data")))
    
    keys = list(set(map(lambda dt: str(dt['omega']), calculated_data)))
    grouped = {f'{k}': list() for k in keys}
    list(map(lambda item: grouped[str(item['omega'])].append(item), tqdm(calculated_data, desc="Grouping by")))
    
    # filtering for the dets that are diferen sings Bolzano theorem
    filtered = list(map(lambda k: 
        list(filter(lambda item: (item[0]['det']*item[1]['det'])<0, zip(grouped[k][:-1], grouped[k][1:])))
    , tqdm(grouped.keys(), desc="Searching Signs")))
    
    # filtering det diff bigger then threshold
    filtered = list(map(lambda k: 
        list(filter(lambda item: abs(abs(item[0]['det']) - abs(item[1]['det'])) < THRESHOLD, zip(grouped[k][:-1], grouped[k][1:])))
    , tqdm(grouped.keys(), desc="Applying Threshold")))
    
    # cleaning empt lists
    filtered = list(filter(lambda ls: len(ls)>0, filtered))
    
    # search for the roots aproximating to zero
    found_omega_kx = dict()
    for lss in tqdm(filtered, desc="Sweeping Omegas"):
        temp = list(map(lambda pair: 
            metodo_bissecao(d_1=pair[0], d_2=pair[1], G1=G1)
        , tqdm(lss, desc=f"Exploring Omega {lss[0][0]['omega']}")))
        found_omega_kx.update({f'{temp[0][1]}': temp})
    
    fim = time.perf_counter()
    tempo_execucao = fim - inicio

    plotting(dados=found_omega_kx, x_lbl=KX_LBL, y_lbl=OMEGA_LBL, time=tempo_execucao, tol=THRESHOLD)