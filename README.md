# Comparing Stable Diffusion 3 Fine-Tuning Methods for Image Generation

Welcome ...

## Getting Started

### Environment Setup

1. We use Miniconda to manage the Python environment. Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
   - To install conda via the command line, you can use the following commands (Linux):
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     chmod +x Miniconda3-latest-Linux-x86_64.sh
     ./Miniconda3-latest-Linux-x86_64.sh
     ```
2. After installing Miniconda, you can create a new environment using the `environment.yml` file provided in the repository.

   ```bash
   conda env create -f environment.yml
   ```

3. Finally, you can activate the environment:

   ```bash
   conda activate csci566_project
   ```

   (**_Note:_** You should see `(csci566_project)` in the terminal prompt after activating the environment)

4. When you are done working, deactivate the environment:
   ```bash
   conda deactivate
   ```

### Adding Dependencies

1. If you need to add a new dependency, you can install it using the following command:

   ```bash
   conda install <package_name>
   ```

2. After installing the package, you can update the `environment.yml` file using:
   ```bash
   conda env export > environment.yml
   ```

## Conventional Commits

Please format commit messages according to Angular's [Conventional Commits](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines) specification.
