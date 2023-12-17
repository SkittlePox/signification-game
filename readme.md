# Readme

## Setting up the environment
Our goal is to run `jaxmarl_testdrive.py`. This is a basic script that you should only be able to run if you have everything installed correctly, which should be the case by the end of this section. I've only tested this out on macOS with an M1 chip.

1. Make a conda environment with `conda create -n siggame python=3.9`
2. Activate the environment with `conda activate siggame`
3. Install jax. If you're on macOS, you can run this if you are using conda:
    ```
    conda install -c conda-forge jaxlib=0.4.19
    conda install -c conda-forge jax
    ```
    Or you can run this if you are using mamba (better):
    ```
    mamba install jaxlib=0.4.19
    mamba install jax
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
5. If you're on mac, you may still struggle to run `jaxmarl_testdrive.py` because you'll get a MuJoCo error. However, we don't really care about MuJoCo, so you can just comment out the following lines in the JaxMARL package (I hate how dirty this is):
    ```
    # In /JaxMARL/jaxmarl/environments/__init__.py comment out line 20
    ...
    from .overcooked import Overcooked, overcooked_layouts
    # from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
    from .hanabi import HanabiGame
    ...
    ```
    ```
    # In /JaxMARL/jaxmarl/registration.py comment out lines 19-23
    ...
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
    SwitchRiddle,
    # Ant,
    # Humanoid,
    # Hopper,
    # Walker2d,
    # HalfCheetah,
    InTheGrid,
    InTheGrid_2p,
    HanabiGame,
    ...
    ```
    This should work for `JaxMARL v0.0.2`. If you're using a different version, you may need to find the lines yourself.
6. We also need torch and torchvision, which can be installed with mamba:
    ```
    mamba install torchvision
    ```
    The conda analog would be something like (untested):
    ```
    conda install -c conda-forge torchvision
    ```
7. We also need jax-dataloader, which can be installed with pip:
    ```
    pip install jax-dataloader
    ```
