# Creating your environment

In previous sections, we installed [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) and configured `pip` to use the ATD2025 Gitlab package registry as an `extra-index-url` using your personal access token.

## Create an environment.yml

In your local development folder, create an `environment.yml` file

```yml title="environment.yml"
name: atd2025
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python>=3.9
  - ipykernel
  - pip
  - pip:
    - atd2025  # Installed from the ATD2025 Gitlab Package Index.
```

## Create your environment

To install all dependencies for `atd2025`, in your terminal and from within your repository folder, run

```console
mamba env create -f environment.yml
```

After creating your environment, you can activate the environment by running:

```
mamba activate atd2025
```

## Install the ATD2025 Package

To install the atd2025 package, from within your repository folder, run

```console
pip install atd2025
```

## (Optional) Create an ipykernel

If you plan to use a Jupyter Notebook environment during the challenge, you can create a kernel based on your `atd2025` environment by doing the following (with your `atd2025` conda environment active)
```console
python -m ipykernel install --name atd2025 --user
```

This step will allow you to use your team's conda environment from within Jupyter notebooks using the `atd2025` kernel.

