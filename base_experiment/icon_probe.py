from torchvision.datasets import MNIST
from flax import linen as nn
from flax.training import train_state
import wandb
import jax
import jax.numpy as jnp
import optax
import numpy as np
from omegaconf import OmegaConf
import hydra
from absl import logging

from utils import to_jax


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_dataset(config):
    mnist_dataset = MNIST('/tmp/mnist/', download=True)
    n_env_imgs = config["ENV_NUM_DATAPOINTS"]
    n_probe_val_imgs = config["PROBE_NUM_DATAPOINTS_VALIDATION"]

    images, labels = to_jax(
        mnist_dataset, num_datapoints=n_env_imgs + n_probe_val_imgs)
    images = images.astype('float32') / 255.0

    images = np.expand_dims(images, -1)
    labels = np.expand_dims(labels, -1)

    env_images = images[:n_env_imgs]
    env_labels = labels[:n_env_imgs]

    probe_val_images = images[n_env_imgs:]
    probe_val_labels = labels[n_env_imgs:]

    train_ds = {"image": env_images, "label": env_labels}
    test_ds = {"image": probe_val_images, "label": probe_val_labels}

    return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(config["LEARNING_RATE"])
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_dataset(config)
    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config["NUM_EPOCHS"] + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config["BATCH_SIZE"], input_rng
        )
        _, test_loss, test_accuracy = apply_model(
            state, test_ds['image'], test_ds['label']
        )

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
            ' test_accuracy: %.2f'
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
            )
        )

        metric_dict = {}

        metric_dict.update({'train_loss': train_loss})
        metric_dict.update({'train_accuracy': train_accuracy})
        metric_dict.update({'test_loss': test_loss})
        metric_dict.update({'test_accuracy': test_accuracy})

        wandb.log(metric_dict)
    return state


@hydra.main(version_base=None, config_path="config", config_name="icon_probe")
def train_probe(config):
    config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )
    train_and_evaluate(config)


if __name__ == "__main__":
    train_probe()
