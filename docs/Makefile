# we make two document, one contain the review of rotation equivariance, the other
# is the generation of tight-binding model
all: AutomaticTB.pdf ReviewRotationEquivariance.pdf

.DELETE_ON_ERROR: %.pdf

# $* can only be used in implicit rule
%.pdf: %.tex
	pdflatex $*.tex
	bibtex $*
	pdflatex $*.tex
	pdflatex $*.tex
	rm $*.aux $*.log $*.out
	rm $*.run.xml $*.bbl $*.blg $*-blx.bib

clean: 
	rm *.aux *.log *.out
	rm *.run.xml *.bbl *.blg *-blx.bib