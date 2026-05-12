import numpy as np

from typing import Union
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
        self.vp = vp
        self.vs = vs
        self.rho = rho
        # Deslocation Field
        self._theta = None
        self._omega = None
        
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
    
    # TODO o grau só varia de 0 a 90
    @theta.setter
    def theta(self, angle) -> None:
        """Setter of the `theta` property

        Args:
            angle (float, optional): Angle of incidence of the wave in degrees.
        """
        assert angle>=0.0 and angle<=360.0, "angle is an invalid value, has to be from 0 to 360"
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
    def k_x(self) -> float: #TODO Precisa forçar que o kx possa ser passado diretamente, pois precisa garantir continuidade
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self.k_l*np.sin(self.theta)
    
    @property
    def k_t(self) -> float:
        """_summary_ Get definitions with the group

        Returns:
            float: _description_
        """
        return self.omega/self.vs
    
    @property
    def k_zt(self) -> Union[np.float64, np.complex128]:#TODO Precisa forçar que o k_zt possa ser passado diretamente, pois precisa garantir continuidade
        """_summary_ Get definitions with the group

        Returns:
            Union[np.float64, np.complex128]: _description_
        """
        return np.emath.sqrt((self.k_t**2) - (self.k_x**2))
    
    @property
    def k_zl(self) -> Union[np.float64, np.complex128]:#TODO Precisa forçar que o k_zl possa ser passado diretamente, pois precisa garantir continuidade
        """_summary_  Get definitions with the group

        Returns:
            Union[np.float64, np.complex128]: _description_
        """
        return np.emath.sqrt((self.k_l**2) - (self.k_x**2))
    
    def __str__(self):
        return f"\t\tMaterial Specifications\nVp={self.vp}\tVs={self.vs}\tRho={self.rho}\n\t\tLamé Constants\nLambda={self.lame_lambda}\tMu={self.lame_mu}\n\t\tDisplacement Fields\nTheta={self.theta}\tOmega={self.omega}\n\t\tKs\nKl={self.k_l}\tKx={self.k_x}\tKt={self.k_t}\tKzt={self.k_zt}\tKzl={self.k_zl}"


# TODO confirmar com o professor o nome correto em inglês
# TODO qual o limiar de theta que define uma onda indo pra cima e para baixo? Entre 0 e 180 é considerado cima e 180 e 360 considerado baixo? Ou existe uma "zona morta" próximo a 180 e 360?
class DisplacementPolarizationMatriz(ABC):
    def __init__(self, medium: Medium):
        super().__init__()
        self.medium = medium
    
    @property
    def A1(self) -> np.array:
        """Polarization matrix of the displacement going up.

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return np.array([
            [self.medium.k_x/self.medium.k_l, -(self.medium.k_zt/self.medium.k_t), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [self.medium.k_zl/self.medium.k_l, self.medium.k_x/self.medium.k_t, 0.0 + 0.0j]
        ], dtype=np.complex128)
    
    @property
    def A2(self) -> np.array:
        """Polarization matrix of the displacement going down.

        Returns:
            np.array: returns a 3x3 `np.array` of `dtype=np.complex128`
        """
        return self.A1*np.array([[1,1,1], [1,1,1], [-1,-1,1]])
    
    def __str__(self):
        return f"A1={self.A1}\nA2={self.A2}"
    
    
if __name__ == "__main__":
    # Medium from the botton
    medium_1 = Medium(vp=1000.0, vs=700.0, rho=2500.0)
    medium_1.theta = 30.0
    medium_1.omega = 100.0
    # Medium from the top
    medium_2 = Medium(vp=2000.0, vs=1500.0, rho=5000.0)
    medium_2.theta = 30.0
    medium_2.omega = 100.0
    # Displacement Matriz fom the botton
    disp_1 = DisplacementPolarizationMatriz(medium=medium_1)
    # Displacement Matriz fom the top
    disp_2 = DisplacementPolarizationMatriz(medium=medium_2)
    pass