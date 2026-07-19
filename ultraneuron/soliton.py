"""Heimburg-Jackson electromechanical soliton (thermodynamic membrane) model.

Heimburg & Jackson 2005 (PNAS 102:9790-9795, "On soliton propagation in
biomembranes and nerves") and Heimburg 2007 (*Thermal Biophysics of
Membranes*) propose that the nerve impulse is an electromechanical
density **soliton** propagating in the lipid membrane, rather than a
purely electrical Hodgkin-Huxley event. The pulse is a transient,
localised excursion of the membrane toward its ordered (gel) phase,
launched near the melting (gel ↔ fluid) phase transition temperature
``T_m``.

The excitability of the membrane is controlled by its lateral area
compressibility κ_A^T. Heimburg's central thermodynamic result is the
proportionality between the **excess heat capacity** Δc_p and the
lateral compressibility,

    κ_A^T(T) = κ_A^{T,0} + (γ_A² · T / A) · Δc_p(T),

so κ_A^T inherits the sharp peak of Δc_p and **diverges as T → T_m**
(Heimburg 2007, §"Compressibility and heat capacity"). A membrane is
maximally soft — and therefore maximally excitable, since a soliton is
cheapest to nucleate where the membrane is most compressible — exactly
at its transition.

Biological membranes sit a few to ~15 °C **above** ``T_m`` at body
temperature (the fluid phase; Heimburg 2007 reports biological-membrane
melting ~10-15 °C below the growth/body temperature, e.g. lung
surfactant ``T_m`` ≈ 30 °C). An external acoustic pressure supplies the
density/area perturbation (volumetric strain ε = p/(ρc²)) that biases
the membrane toward the transition and can nucleate the soliton.

Distinctive, falsifiable sign
-----------------------------
Because ``T_m`` < 37 °C, **heating moves the membrane further from the
transition and reduces compressibility / excitability**. This is the
opposite sign to thermal-excitation hypotheses (where warming the
membrane promotes firing). It is the soliton model's sharpest
experimental discriminator and is preserved exactly by
:meth:`SolitonModel.excitability`.

Scope
-----
This is a closed-form, single-membrane excitability model in the same
tier as :class:`~ultraneuron.nice.NICEModel`: it returns a
scalar excitability for a given acoustic pressure amplitude and local
temperature. It does **not** integrate the Heimburg-Jackson soliton
PDE. The optional :meth:`relative_lateral_sound_speed` exposes the
transition-driven dip in the lateral sound velocity (the kinematic
prerequisite for a stable soliton) but is not required by the scoring
interface.

Units
-----
``T_m``, ``sigma_T`` and the temperature argument are in degrees Celsius.
``rho`` is kg/m³, ``c`` is m/s, pressure is Pa. ``compressibility`` and
``excitability`` are dimensionless; ``acoustic_strain`` is the
dimensionless volumetric strain (condensation) ε = p/(ρc²).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SolitonModel:
    """Heimburg-Jackson thermodynamic-membrane excitability model.

    The membrane's relative lateral area compressibility is modelled as a
    baseline plus an excess term proportional to the excess heat capacity
    Δc_p(T) (Heimburg's c_p ↔ compressibility proportionality):

        κ_rel(T) = (1 + c_p_max · s(T)) / (1 + c_p_max)

    where ``s(T)`` is the normalised c_p peak shape (peak 1 at ``T_m``),
    a Gaussian or Lorentzian in (T − T_m)/σ_T. The result is normalised
    so that ``κ_rel(T_m) = 1`` (maximum) and decays toward
    ``1/(1 + c_p_max)`` far from the transition. ``c_p_max`` is therefore
    the peak-to-baseline excess-compressibility ratio: larger values make
    the peak sharper and the membrane more transition-selective.

    An external acoustic pressure of amplitude ``P_amp`` imposes a
    volumetric strain ε = P_amp/(ρc²) on the membrane. The soliton
    **excitability** is the product of how soft the membrane is (κ_rel)
    and how hard the acoustic field pushes on it (|ε|):

        excitability(P_amp, T) = κ_rel(T) · |P_amp / (ρc²)|

    Parameters
    ----------
    T_m : float
        Membrane melting (gel ↔ fluid) transition temperature [°C].
        Default 30.0 — representative of biological membranes, which melt
        ~10-15 °C below body temperature (Heimburg 2007); e.g. lung
        surfactant ``T_m`` ≈ 30 °C, the system Heimburg & Jackson 2005
        fit. Must be < 37 °C for the distinctive heating-reduces-
        excitability sign to hold.
    sigma_T : float
        Half-width of the c_p / compressibility peak [°C]. Default 3.0.
        For ``peak_shape="gaussian"`` this is the Gaussian standard
        deviation; for ``"lorentzian"`` it is the half-width at half
        maximum. A few degrees reflects the broadened transition of a
        multicomponent biological membrane (Heimburg 2007), much wider
        than a single synthetic lipid.
    c_p_max : float
        Peak excess-compressibility ratio (dimensionless), the amplitude
        of Heimburg's Δc_p-driven compressibility excess relative to the
        non-divergent baseline κ_A^{T,0}. Default 10.0 (peak ≈ 10×
        baseline ⇒ relative compressibility falls to ~0.09 far from
        ``T_m``). Set to 0 to recover a temperature-flat membrane.
    rho : float
        Membrane/tissue mass density [kg/m³]. Default 1000.0 (water-like
        soft tissue; matches the repo FDTD convention RHO0 = 1000).
    c : float
        Small-signal speed of sound [m/s]. Default 1540.0 (soft-tissue
        value; matches the repo FDTD convention C0 = 1540). ``rho·c²`` is
        the adiabatic bulk modulus used for the strain conversion.
    peak_shape : str
        Shape of the normalised c_p peak: ``"gaussian"`` (default) or
        ``"lorentzian"``. Modelling choice; both peak at 1 at ``T_m`` and
        decay monotonically away from it.
    """

    T_m: float = 30.0
    sigma_T: float = 3.0
    c_p_max: float = 10.0
    rho: float = 1000.0
    c: float = 1540.0
    peak_shape: str = "gaussian"

    PRESSURE_VALIDITY_LIMIT: float = 1.0e6
    """Soft upper bound [Pa] on the linear strain conversion.

    Below ~1 MPa the volumetric strain ε = P/(ρc²) ≲ 4×10⁻⁴ stays in the
    linear-acoustics regime where the membrane equation of state is well
    approximated by the constant bulk modulus ρc². Above this drive the
    finite-amplitude (B/A) nonlinearity and the curvature of the
    Heimburg-Jackson density equation of state become first-order, and
    the closed-form excitability scalar should be treated as qualitative.
    """

    BODY_TEMPERATURE_C: float = 37.0
    """Mammalian body temperature [°C], the reference operating point.

    The model's distinctive falsifiable sign (heating reduces
    excitability) holds only while ``T_m`` lies below this value; a
    ``T_m`` at or above body temperature flips the sign and triggers a
    warning in :meth:`__post_init__`.
    """

    def __post_init__(self) -> None:
        if self.T_m >= self.BODY_TEMPERATURE_C:
            warnings.warn(
                f"SolitonModel constructed with T_m={self.T_m:.3g} °C "
                f"≥ body temperature {self.BODY_TEMPERATURE_C:.3g} °C; the "
                "distinctive heating-reduces-excitability sign no longer "
                "holds (biological membranes melt below body temperature; "
                "Heimburg 2007).",
                stacklevel=2,
            )

    def _peak_shape(
        self, T_celsius: NDArray[np.floating] | float
    ) -> NDArray[np.floating]:
        """Normalised excess-heat-capacity peak s(T) (peak 1 at ``T_m``)."""
        x = (np.asarray(T_celsius, dtype=float) - self.T_m) / self.sigma_T
        if self.peak_shape == "gaussian":
            return np.exp(-0.5 * x * x)
        if self.peak_shape == "lorentzian":
            return 1.0 / (1.0 + x * x)
        msg = (
            f"Unknown peak_shape {self.peak_shape!r}; "
            "expected 'gaussian' or 'lorentzian'."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compressibility(
        self, T_celsius: NDArray[np.floating] | float
    ) -> NDArray[np.floating] | float:
        """Relative lateral area compressibility κ_rel(T) [dimensionless].

        Peaks at ``κ_rel(T_m) = 1`` (Heimburg's compressibility divergence
        at the transition) and decays toward ``1/(1 + c_p_max)`` away from
        ``T_m``. This is the excitability weighting: the membrane is
        softest, and a soliton cheapest to nucleate, at ``T_m``.

        Accepts a scalar or array temperature [°C] and returns the same.
        """
        shape = self._peak_shape(T_celsius)
        kappa = (1.0 + self.c_p_max * shape) / (1.0 + self.c_p_max)
        return kappa if kappa.ndim else float(kappa)

    def acoustic_strain(
        self, P_amp_pa: NDArray[np.floating] | float
    ) -> NDArray[np.floating] | float:
        """Volumetric strain (condensation) ε = P/(ρc²) [dimensionless].

        The fractional density/area change imposed on the membrane by a
        linear acoustic pressure of amplitude *P_amp_pa* [Pa]. Linear in
        pressure; signed (follows the sign of the input).
        """
        eps = np.asarray(P_amp_pa, dtype=float) / (self.rho * self.c * self.c)
        return eps if eps.ndim else float(eps)

    def excitability(self, P_amp_pa: float, T_celsius: float) -> float:
        """Soliton excitability scalar [dimensionless].

        ``excitability = compressibility(T) · |acoustic_strain(P_amp)|``
        — the product of membrane softness near the transition and the
        acoustic strain driving it. :func:`soliton_excitation` wraps this
        as the per-case downstream scalar entry point, scoring a case as
        ``SolitonModel().excitability(P_focal_pa, 37.0 + peak_dT_K)``.

        Emits a ``UserWarning`` when ``|P_amp_pa|`` exceeds
        :attr:`PRESSURE_VALIDITY_LIMIT` (1 MPa); the scalar is still
        returned but the linear strain conversion is then only qualitative.

        Rises monotonically with pressure amplitude at fixed temperature.
        With ``T_m`` < 37 °C it **falls** as the temperature is raised
        above body temperature (heating away from ``T_m``), the model's
        distinctive falsifiable sign.

        Parameters
        ----------
        P_amp_pa : float
            Acoustic pressure amplitude at the target [Pa].
        T_celsius : float
            Local absolute membrane temperature [°C] (e.g.
            ``37.0 + peak_dT_K``).

        Returns
        -------
        float
            Dimensionless excitability (≥ 0).
        """
        if abs(float(P_amp_pa)) > self.PRESSURE_VALIDITY_LIMIT:
            warnings.warn(
                f"SolitonModel.excitability called with |P_amp_pa|="
                f"{abs(float(P_amp_pa)):.2e} Pa > validity limit "
                f"{self.PRESSURE_VALIDITY_LIMIT:.2e} Pa; the linear strain "
                "conversion ε = P/(ρc²) is extrapolated and the excitability "
                "scalar should be treated as qualitative.",
                stacklevel=2,
            )
        kappa = self.compressibility(T_celsius)
        eps = self.acoustic_strain(P_amp_pa)
        return float(np.abs(eps) * kappa)

    # ------------------------------------------------------------------
    # Optional: transition-driven lateral sound-speed dip
    # ------------------------------------------------------------------

    def relative_lateral_sound_speed(
        self, T_celsius: NDArray[np.floating] | float
    ) -> NDArray[np.floating] | float:
        """Lateral sound speed relative to its transition minimum.

        For a 2D membrane the lateral sound velocity scales as
        c_lat ∝ 1/√κ_A^T, so it **dips to a minimum at ``T_m``** where the
        compressibility peaks. Returns c_lat(T)/c_lat(T_m) = 1/√κ_rel(T),
        which is 1 at ``T_m`` (minimum) and rises to √(1 + c_p_max) far
        from the transition.

        This velocity dip is the kinematic prerequisite for a stable
        Heimburg-Jackson soliton; it is exposed for diagnostics and is
        not used by :meth:`excitability`.
        """
        kappa = self.compressibility(T_celsius)
        return 1.0 / np.sqrt(kappa)


# ---------------------------------------------------------------------------
# Per-case downstream scalar entry point
# ---------------------------------------------------------------------------


def soliton_excitation(
    P_focal_pa: float,
    peak_dT_K: float = 0.0,
    *,
    baseline_T_C: float = 37.0,
    model: SolitonModel | None = None,
) -> float:
    """Per-case Heimburg-Jackson soliton excitability scalar.

    DOWNSTREAM SCALAR ENTRY POINT for the ``neuromod_parameters`` pipeline
    — the soliton sibling of :func:`~ultraneuron.synapse.synaptic_gain`
    and the astrocyte relay's ``excitation_drive``. Scores a single
    exposure case as

        ξ = SolitonModel().excitability(P_focal_pa, baseline_T_C + peak_dT_K)

    from the focal acoustic pressure amplitude and the focal temperature
    rise, the two per-case quantities the upstream solvers already
    produce (e.g. the rescaled focal PNP and the Pennes ``dT_focus_K``).

    Parameters
    ----------
    P_focal_pa : float
        Focal acoustic pressure amplitude [Pa].
    peak_dT_K : float
        Focal temperature rise above baseline [K]. Default 0.0 (isothermal).
    baseline_T_C : float
        Baseline (body) temperature [°C]. Default 37.0.
    model : SolitonModel, optional
        Pre-configured model; a default :class:`SolitonModel` is used when
        omitted (allowing a parameter sweep to inject a perturbed model).

    Returns
    -------
    float
        Dimensionless soliton excitability ξ ≥ 0. Rises with focal
        pressure; with the default ``T_m`` < body temperature it *falls*
        as ``peak_dT_K`` increases — the model's falsifiable sign.
    """
    m = model if model is not None else SolitonModel()
    return m.excitability(P_focal_pa, baseline_T_C + peak_dT_K)
