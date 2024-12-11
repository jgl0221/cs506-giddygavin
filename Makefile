install:
	@pip install -r requirements.txt

run:
	@jupyter notebook NEW_CS506Final.ipynb

test:
	@python -m unittest discover tests/
