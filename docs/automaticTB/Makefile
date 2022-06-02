# we make two document, one contain the review of rotation equivariance, the other
# is the generation of tight-binding model
all: manuscript.pdf

.DELETE_ON_ERROR: %.pdf

manuscript.pdf: manuscript.tex
	pdflatex manuscript.tex
	bibtex manuscript1-blx.aux
	bibtex manuscript2-blx.aux
	bibtex manuscript
	pdflatex manuscript.tex
	pdflatex manuscript.tex

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