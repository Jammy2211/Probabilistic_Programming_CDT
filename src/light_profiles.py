from src import geometry_profiles as gp

import typing
import numpy as np


class LightProfile(gp.GeometryProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalization: float = 0.1,
        radius: float = 0.6,
    ):
        """
        Abstract base class for a light profile, which describes the emission of a galaxy as a function of radius.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        normalization
            Overall normalization normalisation of the light profile.
        radius
            The circular radius containing half the light of this profile.
        """

        super().__init__(centre=centre, axis_ratio=axis_ratio, angle=angle)

        self.normalization = normalization
        self.radius = radius


class LightDeVaucouleurs(LightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalization: float = 0.1,
        radius: float = 0.6,
    ):
        """
        The De Vaucouleurs light profile often used in Astronomy to represent the bulge of galaxies.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        normalization
            Overall normalization normalisation of the light profile.
        radius
            The circular radius containing half the light of this profile.
        """

        super().__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            angle=angle,
            normalization=normalization,
            radius=radius,
        )

    def image_from_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns the image of the De Vaucouleurs light profile on a grid of Cartesian (y,x) coordinates, which are
        first translated to the profile's reference frame.

        Parameters
        ----------
        grid
            The (y, x) coordinates where the image is computed.
        """
        grid_transformed = self.transform_grid_to_reference_frame(grid=grid)
        grid_elliptical_radii = self.grid_to_elliptical_radii(grid=grid_transformed)

        return self.normalization * np.exp(
            -7.66924 * ((grid_elliptical_radii / self.radius) ** (1.0 / 7.66924) - 1.0)
        )


class LightExponential(LightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalization: float = 0.1,
        radius: float = 0.6,
    ):
        """
        The Exponential light profile often used in Astronomy to represent the disk of galaxies.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        normalization
            Overall normalization normalisation of the light profile.
        radius
            The circular radius containing half the light of this profile.
        """

        super().__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            angle=angle,
            normalization=normalization,
            radius=radius,
        )

    def image_from_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns the image of the light profile on a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates where the image is computed.
        """
        grid_transformed = self.transform_grid_to_reference_frame(grid=grid)
        grid_elliptical_radii = self.grid_to_elliptical_radii(grid=grid_transformed)

        return self.normalization * np.exp(
            -1.67838 * ((grid_elliptical_radii / self.radius) ** (1.0 / 1.67838) - 1.0)
        )
