install:
    pip install -r requirements.txt

run:
    jupyter nbconvert --execute FINAL_CS506Project.ipynb --to notebook --output FINAL_CS506Project_Output.ipynb

test:
    pytest tests/
