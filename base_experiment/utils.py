from torchvision import datasets
import jax.numpy as jnp

def to_jax(dataset, num_datapoints=100):
    images = []
    labels = []
    for i in range(num_datapoints):
        img, label = dataset[i]
        images.append(jnp.array(img, dtype=jnp.float32))
        labels.append(jnp.array(label, dtype=jnp.int32))
    return jnp.array(images), jnp.array(labels)


if __name__ == "__main__":
    # Step 1: Download MNIST Dataset
    mnist = datasets.MNIST(root='/tmp/mnist/', download=True)

    # Step 2: Convert to Jax arrays
    images, labels = to_jax(mnist, num_datapoints=100)
    print(images.shape)
