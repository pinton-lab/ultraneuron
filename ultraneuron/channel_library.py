"""Pre-defined mechanosensitive channel parameters and neuron profiles.

Literature-derived parameter sets for common mechanosensitive ion
channels expressed in neural tissue.  Channel instances returned by
the public API are **copies** — callers may modify them freely without
affecting the library defaults.

References
----------
.. [1] Coste et al., "Piezo1 and Piezo2 are essential components of distinct
       mechanically activated cation channels", Science, 2010.
.. [2] Brohawn et al., "Mechanosensitivity is mediated directly by the
       lipid membrane in TRAAK and TREK1 K+ channels", PNAS, 2014.
.. [3] Honoré, "The neuronal background K2P channels: focus on TREK1",
       Nat Rev Neurosci, 2007.
.. [4] Kang & Kim, "Bhalla-Bhalla model cortical pyramidal neuron", 2024.
.. [5] Oh et al., "Ultrasonic Neuromodulation via Astrocytic TRPA1",
       Current Biology 29:3386-3401, 2019.
.. [6] Karashima et al., "TRPA1 is a Ca²⁺-permeable, non-selective cation
       channel", J Neurosci / PNAS, 2010.
"""

from __future__ import annotations

import copy

from .channels import MechanosensitiveChannel

# ---------------------------------------------------------------------------
# Individual channel definitions
# ---------------------------------------------------------------------------

PIEZO1 = MechanosensitiveChannel(
    name="Piezo1",
    T_half=2.7,  # mN/m — Coste et al. 2010, Lewis & Grandl 2015
    delta_A=18.0,  # nm^2
    g_max=25.0,  # pS
    g_density=0.005,  # mS/cm^2 — sparse expression in neurons
    E_rev=0.0,  # mV — non-selective cation
    tau_0=2.0,  # ms — bell-shaped τ baseline (legacy; superseded by tau_act)
    channel_type="cation",
    description="Non-selective cation channel, rapid inactivation. Coste 2010.",
    tau_inact=12.0,  # ms — Coste et al. 2010 Fig. 2 (inactivation τ ~10-15 ms)
    inact_T_half=4.5,  # mN/m — half-inactivation tension above activation T_half
    inact_delta_A=15.0,  # nm^2 — comparable steepness to activation
    Ca_permeability_fraction=0.10,  # P_Ca/P_Na ≈ 5 → ~10% Ca²⁺ at physiological gradients
    tau_act=2.0,  # ms — Phase-5 fixed activation τ from Coste 2010 (replaces bell-shaped τ_w)
)

TRAAK = MechanosensitiveChannel(
    name="TRAAK",
    T_half=5.0,  # mN/m — Brohawn et al. 2014
    delta_A=12.0,  # nm^2
    g_max=45.0,  # pS
    g_density=0.01,  # mS/cm^2
    E_rev=-90.0,  # mV — K+ reversal
    tau_0=3.0,  # ms — legacy bell-shape baseline; superseded by tau_act
    channel_type="potassium",
    description="Two-pore K+ channel, no inactivation. Brohawn 2014.",
    tau_act=3.0,  # ms — Brohawn 2014 single-channel patch-clamp activation rise
)

TREK1 = MechanosensitiveChannel(
    name="TREK-1",
    T_half=3.5,  # mN/m — Honoré 2007
    delta_A=14.0,  # nm^2
    g_max=100.0,  # pS
    g_density=0.02,  # mS/cm^2 — broadly expressed in cortex
    E_rev=-90.0,  # mV
    tau_0=5.0,  # ms — legacy bell-shape baseline; superseded by tau_act
    channel_type="potassium",
    description="Two-pore K+ channel, temperature/mechano-sensitive. Honoré 2007.",
    tau_act=5.0,  # ms — Honoré 2007 review (mid-range of 3-10 ms reported activation τ)
)

TREK2 = MechanosensitiveChannel(
    name="TREK-2",
    T_half=4.0,  # mN/m
    delta_A=13.0,  # nm^2
    g_max=60.0,  # pS
    g_density=0.015,  # mS/cm^2
    E_rev=-90.0,  # mV
    tau_0=4.0,  # ms — legacy bell-shape baseline; superseded by tau_act
    channel_type="potassium",
    description="Two-pore K+ channel, similar to TREK-1. Kang 2024.",
    tau_act=4.0,  # ms — Kang & Kim 2024 (TREK-2 kinetics, similar to TREK-1)
)

TRPA1 = MechanosensitiveChannel(
    name="TRPA1",
    T_half=6.0,  # mN/m — ESTIMATE; mechanogating tension not directly quantified (Oh 2019)
    delta_A=15.0,  # nm^2 — ESTIMATE; comparable steepness to other MS channels
    g_max=40.0,  # pS — Karashima 2010 (TRPA1 single-channel, tens of pS)
    g_density=0.02,  # mS/cm^2 — astrocytic expression assumption
    E_rev=0.0,  # mV — non-selective cation (Karashima 2010)
    tau_0=3.0,  # ms — legacy bell-shape baseline; superseded by tau_act
    channel_type="cation",
    description=(
        "Astrocytic mechanosensitive Ca²⁺-permeable cation channel; necessary "
        "sensor for ultrasonic neuromodulation via the astrocyte→glutamate→NMDA "
        "relay (Oh et al. 2019). Gate it on membrane tension like the other MS "
        "channels; the slow relay dynamics live in AstrocyteRelay, not here."
    ),
    Ca_permeability_fraction=0.18,  # P_Ca/P_Na ≈ 5–8 → ~15–20% Ca²⁺ (Karashima 2010)
    tau_act=3.0,  # ms — fast channel gating (relay slowness is downstream)
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CHANNEL_REGISTRY: dict[str, MechanosensitiveChannel] = {
    "piezo1": PIEZO1,
    "traak": TRAAK,
    "trek1": TREK1,
    "trek-1": TREK1,
    "trek2": TREK2,
    "trek-2": TREK2,
    "trpa1": TRPA1,
}


def get_channel(name: str) -> MechanosensitiveChannel:
    """Return a copy of a pre-defined channel by name (case-insensitive).

    Parameters
    ----------
    name : str
        Channel name. Accepted values: ``"Piezo1"``, ``"TRAAK"``,
        ``"TREK1"`` / ``"TREK-1"``, ``"TREK2"`` / ``"TREK-2"``.

    Returns
    -------
    MechanosensitiveChannel
        Deep copy of the library channel.

    Raises
    ------
    KeyError
        If *name* does not match any known channel.

    """
    key = name.lower().replace(" ", "")
    if key not in _CHANNEL_REGISTRY:
        available = ", ".join(sorted({ch.name for ch in _CHANNEL_REGISTRY.values()}))
        msg = f"Unknown channel '{name}'. Available: {available}"
        raise KeyError(msg)
    return copy.deepcopy(_CHANNEL_REGISTRY[key])


# ---------------------------------------------------------------------------
# Neuron profiles — typical channel complements per cell type
# ---------------------------------------------------------------------------

NEURON_PROFILES: dict[str, list[MechanosensitiveChannel]] = {
    "cortical_pyramidal": [PIEZO1, TREK1],
    "cortical_interneuron": [PIEZO1, TRAAK],
    "hippocampal_pyramidal": [PIEZO1, TREK1, TREK2],
    "thalamic_relay": [TREK1],
    "cerebellar_purkinje": [PIEZO1, TREK1, TREK2],
    "striatal_msn": [PIEZO1, TREK1],
}


def get_neuron_profile(name: str) -> list[MechanosensitiveChannel]:
    """Return a list of channel copies for a named neuron type.

    Parameters
    ----------
    name : str
        Neuron type. Case-insensitive, underscores and spaces accepted.
        Available: ``"cortical_pyramidal"``, ``"cortical_interneuron"``,
        ``"hippocampal_pyramidal"``, ``"thalamic_relay"``,
        ``"cerebellar_purkinje"``, ``"striatal_msn"``.

    Returns
    -------
    list[MechanosensitiveChannel]
        Deep copies of the channels in the profile.

    Raises
    ------
    KeyError
        If *name* does not match any known profile.

    """
    key = name.lower().replace(" ", "_")
    if key not in NEURON_PROFILES:
        available = ", ".join(sorted(NEURON_PROFILES))
        msg = f"Unknown neuron profile '{name}'. Available: {available}"
        raise KeyError(msg)
    return copy.deepcopy(NEURON_PROFILES[key])
