"""Napari-render config for the micro-CT skull TIME-REVERSAL field.

Renders the two-phase time-reversal wave through the high-res micro-CT skull,
in the same style as the halle_dacc validation-report wave animation:
  * outward  -- virtual point source at the TARGET emits, wavefront expands out
                through the trabecular skull to the transducer aperture, then
  * inward   -- the aperture re-emits the time-reversed signal, refocusing back
                from the source to the target.

Data staged by:
  fullwave2_transcranial_aperturegrowth/export_tr_fields_napari.m  (raw f32) ->
  scripts/assemble_tr_napari.py                                    (-> .npy)
"""

SIM_NAME = "tr_microct"
SIM_TITLE = "Micro-CT skull time reversal (outward target->source, inward source->target) @ 1 MHz"

F0 = 1e6
C0 = 1540.0
CFL = 0.2

SIM_DATA_DIR = "data/tr_microct"
OUT_DIR = "data/tr_microct"
