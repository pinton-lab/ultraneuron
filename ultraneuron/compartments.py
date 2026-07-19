"""Compartmental geometry for multi-compartment HH neurons.

Defines the dendrite-soma-AIS skeleton used by
``MulticompartmentMechanoHHNeuron``.  Each compartment carries its own
ionic-channel densities and a per-channel mechanosensitive-density
scale factor; the three compartments are coupled by axial conductances
parameterised relative to the soma.

Coupling convention (Pinsky-Rinzel style)
-----------------------------------------
Membrane equations at each compartment:

    C_m · dV_s/dt = -I_ion_s - I_mech_s
                    + g_ds · (V_d - V_s)
                    + g_sa · (V_a - V_s) + I_stim_s
    C_m · dV_d/dt = -I_ion_d - I_mech_d
                    + g_ds · (A_s/A_d) · (V_s - V_d) + I_stim_d
    C_m · dV_a/dt = -I_ion_a - I_mech_a
                    + g_sa · (A_s/A_a) · (V_s - V_a) + I_stim_a

where ``g_ds`` and ``g_sa`` are in mS/cm^2 normalised to soma area, and
``A_d, A_s, A_a`` are the dendrite, soma, AIS surface areas.  The area
ratios make the discrete cable equation charge-conserving.

Default literature values (cortical regular-spiking pyramidal cell):
    AIS  Na density:  ~2000 mS/cm^2 (Kole 2008; lower end of 2000–5000
                                     literature range; chosen so explicit RK4
                                     is stable at dt = 0.001 ms while still
                                     giving AIS-first spike initiation)
    Soma Na density:   ~120 mS/cm^2 (Mainen-Sejnowski 1996)
    Dend Na density:    ~30 mS/cm^2 (Stuart & Sakmann 1995)
    Piezo1 dendrite/soma density ratio: ~5 (qualitative; placeholder)

Increasing AIS g_Na to 5000 (Hu 2009 anatomical density) requires
dt ≤ 0.0005 ms — the explicit RK4 integrator becomes the bottleneck
above that density.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Compartment:
    """A single compartment of a multi-compartment HH neuron.

    Parameters
    ----------
    name : str
        Compartment label (``"dendrite"``, ``"soma"``, ``"ais"``).
    g_Na, g_K, g_L : float
        Maximum ionic conductance densities [mS/cm^2].
    E_Na, E_K, E_L : float
        Reversal potentials [mV].
    C_m : float
        Specific membrane capacitance [uF/cm^2]. Default 1.0.
    area : float
        Surface area [cm^2 or arbitrary consistent units]. Only ratios
        between compartments are physically meaningful.
    mechano_density_scale : list[float]
        Per-channel multiplier on the neuron's mechanosensitive channel
        ``g_density`` values (one float per channel in
        ``MulticompartmentMechanoHHNeuron.mechano_channels``).  An
        empty list means "use the channel's bare ``g_density``" (scale = 1).

    """

    name: str
    g_Na: float
    g_K: float
    g_L: float
    E_Na: float = 50.0
    E_K: float = -100.0
    E_L: float = -70.0
    C_m: float = 1.0
    area: float = 1.0
    mechano_density_scale: list[float] = field(default_factory=list)


@dataclass
class CompartmentalGeometry:
    """Three-compartment skeleton: dendrite ↔ soma ↔ AIS.

    Parameters
    ----------
    dendrite, soma, ais : Compartment
        The three compartments.  Order is fixed; the integrator stacks
        state arrays as ``[dendrite, soma, ais]``.
    g_dend_soma : float
        Coupling conductance between dendrite and soma [mS/cm^2 of soma].
    g_soma_ais : float
        Coupling conductance between soma and AIS [mS/cm^2 of soma].

    Notes
    -----
    Coupling is asymmetric in the discrete cable equation: the same
    junction current ``g · ΔV`` divided by compartment area gives a
    different per-area contribution to each side.  See module docstring.

    """

    dendrite: Compartment
    soma: Compartment
    ais: Compartment
    g_dend_soma: float = 1.0
    g_soma_ais: float = 50.0

    def __post_init__(self) -> None:
        # Sanity check: mechano_density_scale lists must all match length
        # (or all be empty) — checked at neuron construction once channels
        # are known. We only check for negative areas here.
        for c in (self.dendrite, self.soma, self.ais):
            if c.area <= 0:
                msg = f"Compartment '{c.name}' must have area > 0, got {c.area}"
                raise ValueError(msg)
        if self.g_dend_soma < 0 or self.g_soma_ais < 0:
            msg = "Coupling conductances must be non-negative."
            raise ValueError(msg)


def cortical_pyramidal_3comp(
    *,
    n_mechano_channels: int = 0,
    dendrite_piezo_scale: float = 5.0,
    ais_piezo_scale: float = 0.0,
) -> CompartmentalGeometry:
    """Default literature-parameterised RS cortical pyramidal geometry.

    Densities follow Kole 2008 (AIS), Mainen-Sejnowski 1996 (soma),
    Stuart & Sakmann 1995 (dendrite).  Mechanosensitive density is
    weighted toward the dendrite per Lewis & Grandl 2015 / Dalghi 2019.

    Parameters
    ----------
    n_mechano_channels : int
        Number of mechanosensitive channels in the parent neuron.
        The returned compartments' ``mechano_density_scale`` lists are
        sized accordingly (uniform scale per compartment).
    dendrite_piezo_scale : float
        Multiplier on each mechano channel's ``g_density`` for the
        dendrite compartment. Default 5.0.
    ais_piezo_scale : float
        Multiplier for the AIS compartment. Default 0.0 (no mechano).

    Returns
    -------
    CompartmentalGeometry

    """
    dend_scale = [dendrite_piezo_scale] * n_mechano_channels
    soma_scale = [1.0] * n_mechano_channels
    ais_scale = [ais_piezo_scale] * n_mechano_channels

    dendrite = Compartment(
        name="dendrite",
        g_Na=30.0,
        g_K=15.0,
        g_L=0.1,
        E_Na=50.0,
        E_K=-100.0,
        E_L=-70.0,
        C_m=1.0,
        area=5.0,  # large surface area
        mechano_density_scale=dend_scale,
    )
    soma = Compartment(
        name="soma",
        g_Na=120.0,
        g_K=36.0,
        g_L=0.1,
        E_Na=50.0,
        E_K=-100.0,
        E_L=-70.0,
        C_m=1.0,
        area=1.0,
        mechano_density_scale=soma_scale,
    )
    ais = Compartment(
        name="ais",
        g_Na=2000.0,
        g_K=80.0,
        g_L=0.1,
        E_Na=50.0,
        E_K=-100.0,
        E_L=-70.0,
        C_m=1.0,
        area=0.05,
        mechano_density_scale=ais_scale,
    )

    return CompartmentalGeometry(
        dendrite=dendrite,
        soma=soma,
        ais=ais,
        g_dend_soma=1.0,
        g_soma_ais=50.0,
    )
