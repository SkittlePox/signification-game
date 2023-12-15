# Readme

## Setting up the environment
1. Make a conda environment with `conda create -n siggame python=3.9`
2. Activate the environment with `conda activate siggame`
3. Install jax. If you're on macOS, you can run this:
    ```
    conda install -c conda-forge jax
    conda install -c conda-forge jaxlib
    ```
    Verify that it is installed correctly by running 
    ```python -c 'import jax; print(jax.numpy.arange(10))'```
    You may get some warnings but you should end up with a list of 0-9.
4. Install JaxMARL. This cannot be done with pip, since we need to run the algorithms (See [JaxMARL installation instructions](https://github.com/FLAIROx/JaxMARL/tree/main?tab=readme-ov-file#installation--) for why).
Clone JaxMARL in a separate directory with:
    ```
    git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
    pip install -e .
    ```
