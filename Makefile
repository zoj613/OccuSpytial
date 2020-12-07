.PHONY: clean test wheels cythonize extras

DOCKER_IMAGES = quay.io/pypa/manylinux1_x86_64 \
	      quay.io/pypa/manylinux2010_x86_64 \
	      quay.io/pypa/manylinux2014_x86_64

define make_wheels
	docker pull $(1) 
	docker container run -t --rm -e PLAT=$(strip $(subst quay.io/pypa/,,$(1))) \
		-v $(shell pwd):/io $(1) /io/build-wheels.sh
endef

clean:
	rm -Rf build/* dist/* occuspytial/*.c occuspytial/*.so occuspytial/*.html \
		occuspytial.egg-info **/*__pycache__ __pycache__ .coverage*

cythonize:
	cythonize occuspytial/*.pyx

sdist: cythonize
	poetry build -f sdist

wheels: clean cythonize
	$(foreach img, $(DOCKER_IMAGES), $(call make_wheels, $(img));)

extras:
	poetry install -E docs

test: extras
	pytest --cov=occuspytial --cov-report=html -v
