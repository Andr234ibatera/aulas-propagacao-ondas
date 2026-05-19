import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional, List
from abc import ABC
from enum import Enum


class Medium(ABC):
    def __init__(self, vp: float, vs: float, rho: float, *args: any, **kwargs: any):
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
        self._theta: float = None
        self._omega: float = None
        # Continuit Values
        self._k_x: float = None
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
    def theta(self) -> float:
        """The angle between wave vectors `k` and position vectors `r` in the scalar product.
        
        Returns:
            float: _description_
        """
        return self._theta
    
    @theta.setter
    def theta(self, angle) -> None:
        """Setter of the `theta` property

        Args:
            angle (float, optional): Angle of incidence of the wave in degrees.
        """
        assert angle>=0.0 and angle<=90.0, "angle is an invalid value, has to be from 0 to 90"
        self._theta = (angle*np.pi)/180
    
    @property
    def omega(self) -> float:
        """Omega represents the angular frequency of the wave.

        Returns:
            float: _description_
        """
        return self._omega
    
    @omega.setter
    def omega(self, frequence) -> None:
        """Setter of the `omega` property
        
        Args:
            frequence (float, optional): Frequence of the wave in Hertz.
        """
        assert frequence>0, "frequance has to be greater then 0"
        self._omega = (2*np.pi)*frequence
    
    @property
    def k_l(self) -> float:
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self.omega/self.vp
    
    @property
    def k_x(self) -> float:
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self._k_x if self._k_x else self.k_l*np.sin(self.theta)
    
    @k_x.setter
    def k_x(self, inherited: float) -> None:
        """_summary_

        Args:
            inherited (Union[np.float64, np.complex128]): _description_
        """
        self._k_x = inherited
    
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
        return f"\t\tMaterial Specifications\nVp={self.vp}\tVs={self.vs}\tRho={self.rho}\n\t\tLamé Constants\nLambda={self.lame_lambda}\tMu={self.lame_mu}\n\t\tDisplacement Fields\nTheta={self.theta}\tOmega={self.omega}\n\t\tKs\nKl={self.k_l}\tKx={self.k_x}\tKt={self.k_t}\tKzt={self.k_zt}\tKzl={self.k_zl}"


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


def lineplot(vetores: List[np.array], thetas: List[float]) -> None:
    angles = list(map(lambda t: (t*180)/np.pi, thetas))
    
    onda_P = np.array([v[0, 0] for v in vetores])
    onda_Sv = np.array([-v[1, 0] for v in vetores])
    onda_Sh = np.array([v[2, 0] for v in vetores])

    # =====================================================================
    # 3. CRIANDO A VISUALIZAÇÃO
    # =====================================================================
    # Criando a figura e os eixos (1 linha, 2 colunas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot da Esquerda: Parte Real ---
    ax1.plot(angles, onda_P.real, label='Onda P', color='blue')
    ax1.plot(angles, onda_Sv.real, label='Onda Sv', color='orange')
    ax1.plot(angles, onda_Sh.real, label='Onda Sh', color='green')
    ax1.set_title('Parte Real da Amplitude das Ondas')
    ax1.set_xlabel('Angulo em Graus')
    ax1.set_ylabel('Amplitude das Ondas')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- Plot da Direita: Parte Imaginária ---
    ax2.plot(angles, onda_P.imag, label='Onda P', color='blue')
    ax2.plot(angles, onda_Sv.imag, label='Onda Sv', color='orange')
    ax2.plot(angles, onda_Sh.imag, label='Onda Sh', color='green')
    ax2.set_title('Parte Imaginária da Amplitude das Ondas')
    ax2.set_xlabel('Angulo em Graus')
    ax2.set_ylabel('Amplitude das Ondas')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Ajusta o layout para que os gráficos não fiquem sobrepostos
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()
    
if __name__ == "__main__":
    medium_data = {
        'm_5': {'vp': 1700.0, 'vs': 1000.0, 'rho': 2000.0, 'h': np.inf},
        'm_4': {'vp': 3500.0, 'vs': 1343.0, 'rho': 3000.0, 'h': 2.0},
        'm_3': {'vp': 5000.0, 'vs': 2000.0, 'rho': 4000.0, 'h': 0.7},
        'm_2': {'vp': 2000.0, 'vs': 1500.0, 'rho': 5000.0, 'h': 1.0},
        'm_1': {'vp': 1000.0, 'vs': 700.0, 'rho': 2500.0, 'h': np.inf},
    }
    
    # Definindo os meios
    k_x = None
    mediums = dict()
    for i in range(5,0,-1):
        medium_temp = list(map(lambda x: 
            Medium(
                vp=medium_data[f'm_{i}']['vp'], 
                vs=medium_data[f'm_{i}']['vs'], 
                rho=medium_data[f'm_{i}']['rho']), range(0, 91)))
        list(map(lambda m, i: m.__setattr__("theta", float(i)), medium_temp, range(0, 91)))
        list(map(lambda m: m.__setattr__("omega", 100.0), medium_temp))
        
        if k_x:
            list(map(lambda m, k: m.__setattr__("k_x", k), medium_temp, k_x))
        else:
            k_x = list(map(lambda m: m.k_x, medium_temp))
        
        mediums.update({f'{i}': medium_temp})
    
    # Definindo as matrizes de deslocamento
    displacements = dict()
    for i in range(5,0,-1):
        displacements.update({f'{i}':
            list(map(lambda m: DisplacementPolarizationMatriz(medium=m), mediums[f'{i}']))})
    
    # L Matriz from the top to bottom
    ls = dict()
    for i in range(5,0,-1):
        ls.update({f'{i}':
            list(map(lambda m, d: LMatriz(medium=m, displacement=d), mediums[f'{i}'], displacements[f'{i}']))})
    
    # Local Impedance Tensor from the top to bottom
    zs = dict()
    for i in range(5,0,-1):
        zs.update({f'{i}':
            list(map(lambda m, d, l: LocalImpedanceTensor(medium=m, displacement=d, l_matriz=l), mediums[f'{i}'], displacements[f'{i}'], ls[f'{i}']))})
    
    # Defining Ms Matriz
    
    
    # Defining Gs
    # Gs = dict()
    # for i in range(1,4):
    #     if len(Gs) > 0:
            
    #         # temp = zs[str(i)].Z1@
    #     else:
    #         Gs.update({str(i): zs[str(i)].Z2})
    
    pass    
    
    # Rs = list(map(lambda z2: np.linalg.inv((z2.Z1 - G))@(G - z2.Z2), zs_2))
    
    # C_p = 1.0 + 0.0j #(amplitude onda C_p=1)
    # Us_2 = list(map(lambda m: np.array([
    #     [C_p*np.sin(m.theta)], [0.0 + 0.0j], [-(C_p*np.cos(m.theta))]
    # ], dtype=np.complex128), mediums_2))

    # Us_1 = list(map(lambda r, u2: r@u2, Rs, Us_2))
    
    # Cps_1 = list(map(lambda u, d: np.linalg.inv(d.A1)@u, Us_1, disps_2))
    
    # # TODO Levando em consideração C_alpha = (A_alpha)^-1@U_alpha plotar o C_alpha para o meio superior
    
    # lineplot(vetores=Cps_1, thetas=list(map(lambda m: m.theta, mediums_2)))
    
    pass