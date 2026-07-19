"""Neuromodulation pipeline: strain -> membrane tension -> neural firing.

Connects tissue-level elastomechanics (e.g. from a shear wave FDTD solver)
to neural firing predictions via experimentally parameterized
mechanosensitive ion channels and a modified Hodgkin-Huxley neuron model.
"""

from .astrocyte import AstrocyteRelay, simulate_astrocyte_relay
from .channel_library import (
    NEURON_PROFILES,
    PIEZO1,
    TRAAK,
    TREK1,
    TREK2,
    TRPA1,
    get_channel,
    get_neuron_profile,
)
from .calcium import CalciumDynamics, KCaChannel
from .channels import MechanosensitiveChannel, strain_to_membrane_tension
from .nmda import NMDAReceptor, nmdar_stretch_factor
from .compartments import Compartment, CompartmentalGeometry, cortical_pyramidal_3comp
from .multicompartment import (
    MulticompartmentMechanoHHNeuron,
    simulate_map_3comp,
    simulate_map_3comp_with_spike_times,
)
from .neuron import (
    K_MECHANO_NAV,
    MechanoHHNeuron,
    bioheat_to_absolute_temperature,
    nav_current_gain,
    simulate_map,
)
from .synapse import MechanoSynapse, simulate_synaptic_relay, synaptic_gain
from .nice import NICEModel
from .provenance import (
    SOLITON_PROVENANCE,
    ParamSource,
    Provenance,
    format_provenance_report,
    provenance_for,
    registered_names,
)
from .soliton import SolitonModel, soliton_excitation
from .strain import compute_strain_invariants, compute_strain_tensor
from .thermosensors import TRPV1, TRPV4, TRPThermosensorChannel

__all__ = [
    "AstrocyteRelay",
    "CalciumDynamics",
    "Compartment",
    "CompartmentalGeometry",
    "K_MECHANO_NAV",
    "KCaChannel",
    "MechanoHHNeuron",
    "MechanoSynapse",
    "MechanosensitiveChannel",
    "MulticompartmentMechanoHHNeuron",
    "NEURON_PROFILES",
    "NICEModel",
    "NMDAReceptor",
    "PIEZO1",
    "TRAAK",
    "TREK1",
    "TREK2",
    "TRPA1",
    "ParamSource",
    "Provenance",
    "SOLITON_PROVENANCE",
    "SolitonModel",
    "TRPThermosensorChannel",
    "TRPV1",
    "TRPV4",
    "format_provenance_report",
    "provenance_for",
    "registered_names",
    "bioheat_to_absolute_temperature",
    "compute_strain_invariants",
    "compute_strain_tensor",
    "cortical_pyramidal_3comp",
    "get_channel",
    "get_neuron_profile",
    "nav_current_gain",
    "nmdar_stretch_factor",
    "simulate_astrocyte_relay",
    "simulate_map",
    "simulate_map_3comp",
    "simulate_map_3comp_with_spike_times",
    "simulate_synaptic_relay",
    "soliton_excitation",
    "strain_to_membrane_tension",
    "synaptic_gain",
]
