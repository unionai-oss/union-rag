.PHONY: install-jupytext
install-jupytext:
	pip install jupyter jupytext

.PHONY: notebook
notebook: install-jupytext
	jupytext --to ipynb workshop.qmd
