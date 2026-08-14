TEX = main.tex

.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode $(TEX)

clean:
	latexmk -C $(TEX)
