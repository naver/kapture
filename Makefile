.PHONY: install install_dev docker clean check-docker check-asciidoctor check-pandoc unittests

build: README.md
	python3 -m build

install: README.md
	pip install .

install_dev: README.md
	pip install -e .

docker: check-docker
	docker build -f docker/Dockerfile . -t kapture/kapture

check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "docker is not installed"; exit 1; }

README.md: README.adoc
	asciidoctor -b docbook $< -o - | pandoc -f docbook -t gfm -o $@

check-asciidoctor:
	@command -v asciidoctor >/dev/null 2>&1 || { echo "asciidoctor is not installed"; exit 1; }

check-pandoc:
	@command -v pandoc >/dev/null 2>&1 || { echo "pandoc is not installed"; exit 1; }

clean:
	rm -rf dist/ kapture.egg-info/ README.md

all: install

unittests:
	python3 -m unittest discover -s tests