import numpy as np
from typing import Optional, List


class Galaxy:
    def __init__(
        self,
        redshift: float,
        light_profiles: Optional[List] = None,
        mass_profiles: Optional[List] = None,
    ):
        """
        A galaxy, which contains light and mass profiles at a specified redshift.

        Parameters
        ----------
        redshift
            The redshift of the galaxy.
        light_profiles
            A list of the galaxy's light profiles.
        mass_profiles
            A list of the galaxy's mass profiles.
        """

        self.redshift = redshift
        self.light_profiles = light_profiles
        self.mass_profiles = mass_profiles

    def image_from_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns the summed 2D image of all of the galaxy's light profiles using an input
        grid of Cartesian (y,x) coordinates.

        If the galaxy has no light profiles, a grid of zeros is returned.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.light_profiles is not None:
            return sum(map(lambda p: p.image_from_grid(grid=grid), self.light_profiles))
        return np.zeros((grid.shape[0],))

    def deflections_from_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns the summed (y,x) deflection angles of the galaxy's mass profiles
        using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, two grid of zeros are returned.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.mass_profiles is not None:
            return sum(
                map(lambda p: p.deflections_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0], 2))


class Redshift(float):
    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
