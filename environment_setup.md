# Environment Setup

## Setting up the local dev environment with pipenv

The tensorflow version available to me locally was `2.16.2` and the Python version is 3.11.

1. Install pipenv

2. Create environment with pipenv

```bash
pipenv install tensorflow pillow pandas==2.2.3
pipenv install --dev matplotlib jupyter ipykernel
```

3. Register Kernel with Jupyter 

```bash
pipenv shell

python -m ipykernel install --user --name=emotion_recog --display-name "emotion_recog"
```

4. Start jupyter notebook

```bash
pipenv shell
jupyter notebook
```
