install:
	@pip install -r requirements.txt

run:
	@jupyter notebook FINAL_CS506Project.ipynb

test:
	@python -m unittest discover tests/
