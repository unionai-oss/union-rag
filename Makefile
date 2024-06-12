.PHONY: install-jupytext
install-jupytext:
	pip install jupyter jupytext

.PHONY: notebook
notebook: install-jupytext
	jupytext --to ipynb workshop.qmd

.PHONY: install-uv
install-uv:
	pip install uv

requirements.lock.txt: install-uv
	uv pip compile requirements.txt -o requirements.lock.txt

.PHONY: workshop-requirements.lock.txt
workshop-requirements.lock.txt: install-uv
	uv pip compile _workshop/requirements.txt -o _workshop/requirements.lock.txt
