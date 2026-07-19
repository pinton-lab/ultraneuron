# Validation-report build targets.
#
# Usage:
#   make report SIM=halle_bowl                    # full report
#   make report SIM=halle_bowl SECTIONS=acoustic_inputs
#   make report SIM=halle_dacc ANIM=1             # + embed 3-D wave animation
#   make tex SIM=halle_bowl                       # render only, no pdflatex
#
# Run from the repo root.

PYTHON      ?= python3
SIM         ?= halle_bowl
SECTIONS    ?= acoustic_inputs
ANIM        ?=
THIS_DIR    := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
CONFIG      := $(THIS_DIR)configs/$(SIM).py
BUILDER     := $(THIS_DIR)report_builder.py
ANIM_FLAG   := $(if $(ANIM),--with-animation,)

.PHONY: report tex clean

report:
	$(PYTHON) $(BUILDER) --config $(CONFIG) --sections $(SECTIONS) $(ANIM_FLAG)

tex:
	$(PYTHON) $(BUILDER) --config $(CONFIG) --sections $(SECTIONS) --no-compile

clean:
	@find . -name 'validation_report' -type d -exec rm -rf {} + 2>/dev/null || true
