import os
from torchvision.datasets import MNIST
from flax import linen as nn
from flax.training import train_state, orbax_utils
import orbax.checkpoint
import wandb
import jax
import jax.numpy as jnp
import optax
import numpy as np
from omegaconf import OmegaConf
import hydra
from absl import logging
import uuid
import yaml
import pathlib
import pickle
from utils import to_jax


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
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
    return logits, grads, loss, accuracy


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
        logits, grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return logits, state, train_loss, train_accuracy


def get_dataset(config):
    if config["ENV_DATASET"] == "mnist":
        mnist_dataset = MNIST('/tmp/mnist/', download=True)
        n_env_imgs = config["ENV_NUM_DATAPOINTS"]
        n_probe_val_imgs = config["PROBE_NUM_DATAPOINTS_VALIDATION"]

        images, labels = to_jax(
            mnist_dataset, num_datapoints=n_env_imgs + n_probe_val_imgs)
        images = images.astype('float32') / 255.0

        images = np.expand_dims(images, -1)

        env_images = images[:n_env_imgs]
        env_labels = labels[:n_env_imgs]

        probe_val_images = images[n_env_imgs:]
        probe_val_labels = labels[n_env_imgs:]

        train_ds = {"image": env_images, "label": env_labels}
        test_ds = {"image": probe_val_images, "label": probe_val_labels}

        return train_ds, test_ds
    
    elif config["ENV_DATASET"] == "cifar10":
        download_path = '/tmp/cifar10/'
        os.makedirs(download_path, exist_ok=True)
        
        # Check if dataset already exists
        dataset_files = os.listdir(download_path)
        if len(dataset_files) > 0:
            print(f"Dataset already exists in {download_path}.")
        else:
            dataset_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            
            import requests
            from tqdm import tqdm

            response = requests.get(dataset_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(os.path.join(download_path, 'cifar-10.tar.gz'), 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("Failed to download the dataset.")
            else:
                print("Dataset downloaded successfully.")
                import tarfile
                tar = tarfile.open(os.path.join(download_path, 'cifar-10.tar.gz'), 'r:gz')
                tar.extractall(download_path)
                tar.close()

        with open(download_path+'cifar-10-batches-py/data_batch_1', 'rb') as f:   # Each batch is 10,000 images. Probably don't need that many
            data_batch_1 = pickle.load(f, encoding='bytes')
        
        with open(download_path+'cifar-10-batches-py/data_batch_2', 'rb') as f:
            data_batch_2 = pickle.load(f, encoding='bytes')

        with open(download_path+'cifar-10-batches-py/data_batch_3', 'rb') as f:
            data_batch_3 = pickle.load(f, encoding='bytes')

        raw_images_1 = data_batch_1[b'data'].astype('float32') / 255.0
        raw_images_2 = data_batch_2[b'data'].astype('float32') / 255.0
        raw_images_3 = data_batch_3[b'data'].astype('float32') / 255.0

        raw_images = np.vstack((raw_images_1, raw_images_2, raw_images_3))

        red_channel = raw_images[:, :1024].reshape(-1, 32, 32)
        green_channel = raw_images[:, 1024:2048].reshape(-1, 32, 32)
        blue_channel = raw_images[:, 2048:].reshape(-1, 32, 32)

        # Convert to grayscale using the weighted formula: 0.299*R + 0.587*G + 0.114*B
        all_images = jnp.array((0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel), dtype=jnp.float32).reshape(-1, 32, 32, 1)
        
        raw_labels_1 = jnp.array(data_batch_1[b'labels'], dtype=jnp.int32)
        raw_labels_2 = jnp.array(data_batch_2[b'labels'], dtype=jnp.int32)
        raw_labels_3 = jnp.array(data_batch_3[b'labels'], dtype=jnp.int32)
        all_labels = jnp.hstack((raw_labels_1, raw_labels_2, raw_labels_3))

        env_images = all_images[:config["ENV_NUM_DATAPOINTS"]]
        env_labels = all_labels[:config["ENV_NUM_DATAPOINTS"]]

        probe_val_images = all_images[config["ENV_NUM_DATAPOINTS"]:config["ENV_NUM_DATAPOINTS"]+config["PROBE_NUM_DATAPOINTS_VALIDATION"]]
        probe_val_labels = all_labels[config["ENV_NUM_DATAPOINTS"]:config["ENV_NUM_DATAPOINTS"]+config["PROBE_NUM_DATAPOINTS_VALIDATION"]]

        train_ds = {"image": env_images, "label": env_labels}
        test_ds = {"image": probe_val_images, "label": probe_val_labels}

        return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, config["ENV_KWARGS"]["image_dim"], config["ENV_KWARGS"]["image_dim"], 1]))['params']
    if config["OPTIMIZER"] == "sgd":
        tx = optax.sgd(config["LEARNING_RATE"], config["MOMENTUM"])
    elif config["OPTIMIZER"] == "adam":
        tx = optax.adam(config["LEARNING_RATE"])
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def calculate_entropy(logits, labels=None):
    softmax_probs = jax.nn.softmax(logits, axis=-1)
    entropies = -jnp.sum(softmax_probs * jnp.log(softmax_probs + 1e-9), axis=-1)
    mean_entropy = jnp.mean(entropies)

    if labels is not None:
        unique_labels = jnp.unique(labels)
        per_class_entropy = jnp.array([jnp.mean(entropies[labels == c]) for c in unique_labels])
        return mean_entropy, per_class_entropy
        # NOTE: This entropy calculation is likely wrong but I don't know why. It doesn't matter because we will only report mean_entropy in the paper, which I believe is calculated correctly.

    return mean_entropy


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
        train_logits, state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config["BATCH_SIZE"], input_rng
        )
        test_logits, _, test_loss, test_accuracy = apply_model(
            state, test_ds['image'], test_ds['label']
        )

        train_entropy, train_per_class_entropy = calculate_entropy(train_logits, train_ds['label'][epoch*config["BATCH_SIZE"]:(epoch+1)*config["BATCH_SIZE"]])
        test_entropy, test_per_class_entropy = calculate_entropy(test_logits, test_ds['label'])

        train_entropy_str = ', '.join([f'{e:.4f}' for e in train_per_class_entropy])
        test_entropy_str = ', '.join([f'{e:.4f}' for e in test_per_class_entropy])

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, train_entropy: %.4f, train_entropy_per_class: [%s],'
            ' test_loss: %.4f, test_accuracy: %.2f, test_entropy: %.4f, test_entropy_per_class: [%s]'
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                train_entropy,
                train_entropy_str,
                test_loss,
                test_accuracy * 100,
                test_entropy,
                test_entropy_str
            )
        )

        metric_dict = {}

        metric_dict.update({'train_loss': train_loss})
        metric_dict.update({'train_accuracy': train_accuracy})
        metric_dict.update({'entropy/train_avg': train_entropy})
        metric_dict.update({'test_loss': test_loss})
        metric_dict.update({'test_accuracy': test_accuracy})
        metric_dict.update({'entropy/test_avg': test_entropy})

        
        for i in range(0, config["ENV_KWARGS"]["num_classes"]):
            metric_dict.update({f'entropy/train_{i}': train_per_class_entropy[i]})
            metric_dict.update({f'entropy/test_{i}': test_per_class_entropy[i]})

        wandb.log(metric_dict)
    return state

def evaluate_model(state, config):
    train_ds, test_ds = get_dataset(config)

    train_logits, _, train_loss, train_accuracy = apply_model(
            state, train_ds['image'], train_ds['label']
        )

    test_logits, _, test_loss, test_accuracy = apply_model(
            state, test_ds['image'], test_ds['label']
        )
    

    train_entropy, train_per_class_entropy = calculate_entropy(train_logits, train_ds['label'])
    test_entropy, test_per_class_entropy = calculate_entropy(test_logits, test_ds['label'])

    train_entropy_str = ', '.join([f'{e:.4f}' for e in train_per_class_entropy])
    test_entropy_str = ', '.join([f'{e:.4f}' for e in test_per_class_entropy])
    
    
    logging.info(
            'train_loss: %.4f, train_accuracy: %.2f, train_entropy: %.4f, train_entropy_per_class: [%s],'
            ' test_loss: %.4f, test_accuracy: %.2f, test_entropy: %.4f, test_entropy_per_class: [%s]'
            % (
                train_loss,
                train_accuracy * 100,
                train_entropy,
                train_entropy_str,
                test_loss,
                test_accuracy * 100,
                test_entropy,
                test_entropy_str
            )
        )
    

def load_model(checkpoint_name, config, no_train=False):
    if no_train:
        config = dict({"OPTIMIZER": "sgd", "MOMENTUM": 0.9, "LEARNING_RATE": 0.0001})
    empty_state = create_train_state(jax.random.key(0), config)
    empty_checkpoint = {'model': empty_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(checkpoint_name, item=empty_checkpoint)
    return raw_restored


def save_model(train_state, config, model_name):
    checkpoint = {'model': train_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(checkpoint)
    local_path = str(pathlib.Path().resolve())
    orbax_checkpointer.save(local_path+'/models/'+model_name, checkpoint, save_args=save_args)
    with open(local_path+'/models/'+model_name+'/config.yaml', 'w') as f:
        yaml.dump(config, f)


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
    
    if config["TRAIN_MODEL"]:
        train_state = train_and_evaluate(config)

        if config["SAVE_MODEL"]:
            random_uuid = uuid.uuid4()
            model_name = config["MODEL_NAME_PREFIX"]
            model_name += str(random_uuid)[-4:]
            save_model(train_state, config, model_name)
    
    if config["EVAL_MODEL"]:
        local_path = str(pathlib.Path().resolve())
        raw_restored = load_model(local_path+'/models/'+config["MODEL_NAME_EVAL"], config)
        train_state = raw_restored['model']

        evaluate_model(train_state, config)


if __name__ == "__main__":
    train_probe()
