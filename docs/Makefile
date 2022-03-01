all: 3dnn structure_representation

structure_representation: structure_representation.pdf
3dnn: 3dnn.pdf

.DELETE_ON_ERROR: %.pdf

# $* can only be used in implicit rule
%.pdf: %.tex
	pdflatex $*.tex
	pdflatex $*.tex
	bibtex $*
	pdflatex $*.tex
	rm $*.aux $*.log $*.out
	rm $*.run.xml $*.bbl $*.blg $*-blx.bib

clean: 
	rm *.aux *.log *.out
	rm *.run.xml *.bbl *.blg *-blx.bib