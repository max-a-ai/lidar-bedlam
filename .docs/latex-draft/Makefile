# Build the Eurographics paper with latexmk.
# The vendored EG style (egpubl.cls, dfadobe.sty, eg-alpha-doi.bst, ...) lives in eg-style/egPublStyle-cgf.
STYLE_DIR := eg-style/egPublStyle-cgf
export TEXINPUTS := .:$(STYLE_DIR)//:
export BSTINPUTS := .:$(STYLE_DIR)//:
export BIBINPUTS := .:

MAIN := main

.PHONY: all clean distclean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
