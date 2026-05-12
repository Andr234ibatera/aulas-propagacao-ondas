import numpy as np

from typing import Union, Optional
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
        
        
    @property
    def lame_mu(self) -> float:
        """The Lamé constants are two independent elastic parameters, `lambda` and `mu`, used to describe the mechanical properties of isotropic materials in wave propagation.
        
        `Mu` is the second Lamé parameter, also called the shear modulus or modulus of rigidity.
        
        Obs: In fluids, where rigidity vanishes `mu = 0`, only lambda governs acoustic wave propagation. In solids, both constants are essential for characterizing stress, strain, and wave behavior. 

        Returns:
            float: returns the second Lamé parameter, also called the shear modulus or modulus of rigidity.
        """
        return (self.vs**2)*self.rho
    
    @property
    def lame_lambda(self) -> float:
        """The Lamé constants are two independent elastic parameters, `lambda` and `mu`, used to describe the mechanical properties of isotropic materials in wave propagation.
        
        `Lambda` is known as the first Lamé parameter and represents pure incompressibility.
        
        Obs: In fluids, where rigidity vanishes `mu = 0`, only lambda governs acoustic wave propagation. In solids, both constants are essential for characterizing stress, strain, and wave behavior.

        Returns:
            float: returns the first Lamé parameter and represents pure incompressibility.
        """
        return ((self.vp**2)*self.rho) - (2*self.lame_mu)
    
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

    
if __name__ == "__main__":
    # Medium from the top
    medium_2 = Medium(vp=2000.0, vs=1500.0, rho=5000.0)
    medium_2.theta = 30.0
    medium_2.omega = 100.0
    # Medium from the botton
    medium_1 = Medium(vp=1000.0, vs=700.0, rho=2500.0)
    medium_1.theta = 30.0
    medium_1.omega = 100.0
    medium_1.k_x = medium_2.k_x
    # Displacement Matriz from the top
    disp_2 = DisplacementPolarizationMatriz(medium=medium_2)
    # Displacement Matriz from the botton
    disp_1 = DisplacementPolarizationMatriz(medium=medium_1)
    # L Matriz from the top
    l_2 = LMatriz(medium=medium_2, displacement=disp_2)
    # L Matriz from the botton
    l_1 = LMatriz(medium=medium_1, displacement=disp_1)
    # Local Impedance Tensor from the top
    z_2 = LocalImpedanceTensor(medium=medium_2, displacement=disp_2, l_matriz=l_2)
    # Local Impedance Tensor from the botton
    z_1 = LocalImpedanceTensor(medium=medium_1, displacement=disp_1, l_matriz=l_1)
    
    # TODO esperar a próxima aula para identificar onde colocar G e R, talvez pode ser definido dentro de uma classe com arrays de camadas
    # Tensor de impedância superficial
    G = z_2.Z2
    # Tensor de reflexão
    R = np.linalg.inv((z_1.Z1 - G))@(G - z_1.Z2)
    pass