# Build the paper.  `make` imports the figures from the simulation, generates
# the language-agent placeholders, and compiles.
#
#   make            build main.pdf
#   make figures    re-import figures only
#   make sim        regenerate the simulation figures, then import them
#   make clean      remove build artefacts (keeps main.pdf)
#   make distclean  also remove main.pdf and imported figures

SIM      := ../nn-based-simulation
SIM_FIGS := $(SIM)/figures/iclr
STYLE    := iclr
PRESET   := medium

# Figures pulled in from the simulation, by the name the .tex files use.
IMPORTS := modulation_contours modulation_contours_all modulation_slices learning_flows polarization \
           order_parameter_maps correlation_maps frustration_maps \
           phase_diagram phase_diagram_large_agenda agenda_trajectories \
           sign_convention_comparison

.PHONY: all figures sim clean distclean watch

all: main.pdf

main.pdf: main.tex references.bib figures
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	@mkdir -p figures
	@for f in $(IMPORTS); do \
		if [ -f "$(SIM_FIGS)/$$f.pdf" ]; then \
			cp "$(SIM_FIGS)/$$f.pdf" figures/; \
		else \
			echo "missing: $(SIM_FIGS)/$$f.pdf  (run 'make sim')"; \
		fi; \
	done
	@echo "figures imported from $(SIM_FIGS)"

# Regenerate the simulation figures at ICLR proportions, then import them.
sim:
	cd $(SIM) && python3 scripts/make_all.py --preset $(PRESET) --style $(STYLE)
	cd $(SIM) && python3 scripts/order_parameter_maps.py --preset $(PRESET) --style $(STYLE)
	cd $(SIM) && python3 scripts/polarization.py --preset $(PRESET) --style $(STYLE)
	cd $(SIM) && python3 scripts/sign_convention_comparison.py --preset $(PRESET) --style $(STYLE)
	$(MAKE) figures

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -c
	rm -f main.aux main.bbl main.blg

distclean: clean
	rm -f main.pdf
	rm -rf figures
