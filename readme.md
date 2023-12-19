# Readme

## Setting up the environment
Our goal is to run `jaxmarl_testdrive.py`. This is a basic script that you should only be able to run if you have everything installed correctly, which should be the case by the end of this section. I've only tested this out on macOS with an M1 chip.

1. Make a conda environment with `conda create -n siggame python=3.9`
2. Activate the environment with `conda activate siggame`
3. Install jax. If you're on macOS, you can run this if you are using conda:
    ```shell
    conda install -c conda-forge jaxlib=0.4.19
    conda install -c conda-forge jax
    ```
    Or you can run this if you are using mamba (better):
    ```shell
    mamba install jaxlib=0.4.19
    mamba install jax
    ```
    Verify that it is installed correctly by running 
    ```python -c 'import jax; print(jax.numpy.arange(10))'```
    You may get some warnings but you should end up with a list of 0-9.
4. Install JaxMARL. This cannot be done with pip, since we need to run the algorithms (See [JaxMARL installation instructions](https://github.com/FLAIROx/JaxMARL/tree/main?tab=readme-ov-file#installation--) for why).
Clone JaxMARL in a separate directory with:
    ```shell
    git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
    pip install -e .
    ```
5. If you're on mac, you may still struggle to run `jaxmarl_testdrive.py` because you'll get a MuJoCo error. However, we don't really care about MuJoCo, so you can just comment out the following lines in the JaxMARL package (I hate how dirty this is):
    ```python
    # In /JaxMARL/jaxmarl/environments/__init__.py comment out line 20
    ...
    from .overcooked import Overcooked, overcooked_layouts
    # from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
    from .hanabi import HanabiGame
    ...
    ```
    ```python
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
    ```shell
    mamba install torchvision
    ```
    The conda analog would be something like (untested):
    ```shell
    conda install -c conda-forge torchvision
    ```
7. We also need jax-dataloader, which can be installed with pip:
    ```shell
    pip install jax-dataloader
    ```

## SimplifiedSignificationGame
The environment setup is based on the State class:
```python
class State:
    # Newly generated, to be evaluated next state after speakers generate images
    next_channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    next_env_images: chex.Array  # [image_size] * num_channels
    next_env_labels: chex.Array  # [num_classes] * num_channels
    next_speaker_labels: chex.Array  # [num_classes] * num_speakers

    # Current state, to be evaluated this state
    channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    env_images: chex.Array  # [image_size] * num_channels
    env_labels: chex.Array  # [num_classes] * num_channels
    speaker_labels: chex.Array  # [num_classes] * num_speakers
    speaker_images: chex.Array  # [image_size] * num_speakers

    # Previous state
    previous_channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    previous_env_images: chex.Array  # [image_size] * num_channels
    previous_env_labels: chex.Array  # [num_classes] * num_channels
    previous_speaker_labels: chex.Array  # [num_classes] * num_speakers
    previous_speaker_images: chex.Array  # [image_size] * num_speakers

    iteration: int
```
In each state, the environment has a foot in two rounds of the game due to the asynchronous nature of the speaker and listener. The speakers generate new images based on the classes sampled at the current time step. These images are then classified by the listeners in the next state. 

To manage these two consecutive rounds, all variables prefixed with `next_` are generated at the current time step for the next round, while all variables without a prefix are for the current round. Variables prefixed with `previous_` were generated in the previous time step. 

The environment consists of two sets of communication channels. One channel contains images from the environment (`env_images`) and the other contains images from the speakers (`speaker_images`). The true labels for these images are stored in `env_labels` and `speaker_labels` respectively. 

The crucial variable is `channel_map`, which is a 3D array of shape `[num_speakers + num_channels] * [num_listeners] * num_channels`. Each index corresponds to a channel, which is a 2D array of `[speaker_index listener_index]`, i.e. a speaker and a listener. The listener's job is to classify the speaker's image. Both the speaker and listener are rewarded (or punished) based on the listener's classification.
