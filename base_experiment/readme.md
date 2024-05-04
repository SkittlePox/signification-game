# Base Experiment

## A brief explanation of `SimplifiedSignificationGame`

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


## Running the base experiment

Running `test_mnist_signification_game()` is a good way to peer into the environment and see how it works. It runs a signification game with the following parameters:
```python
num_speakers = 5
num_listeners = 10
num_channels = 10
num_classes = 10
```

Here's an example (Note that you may not get the same labels as this example due to the dataloader (Idk how to set its seed)):

```
Action for listener_0: 4
Action for listener_1: 9
Action for listener_2: 4
Action for listener_3: 9
Action for listener_4: 0
Action for listener_5: 1
Action for listener_6: 2
Action for listener_7: 8
Action for listener_8: 1
Action for listener_9: 1
channel_map:
[[10  5]
 [ 2  2]
 [ 6  1]
 [ 0  0]
 [ 8  8]
 [12  7]
 [ 4  3]
 [ 1  9]
 [ 3  4]
 [ 5  6]]
speaker_labels:
[2 3 5 0 9]
env_labels:
[2 8 8 9 7 4 6 5 8 0]
Reward for __all__: -5.0
Reward for listener_0: -1.0
Reward for listener_1: -1.0
Reward for listener_2: -1.0
Reward for listener_3: 1.0
Reward for listener_4: 1.0
Reward for listener_5: -1.0
Reward for listener_6: 1.0
Reward for listener_7: -1.0
Reward for listener_8: -1.0
Reward for listener_9: -1.0
Reward for speaker_0: -1.0
Reward for speaker_1: -1.0
Reward for speaker_2: -1.0
Reward for speaker_3: 1.0
```

## Running the full experiment
`kaggle datasets download -d misrakahmed/vegetable-image-dataset`
