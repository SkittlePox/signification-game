import cloudpickle
import pathlib
import sys
import jax.numpy as jnp

local_path = str(pathlib.Path().resolve())+"/base_experiment"
sys.path.append(local_path)
from agents import *

filename = "agents-7b40"
agent_count = 2
listener_agents = []
speaker_agents = []

for i in range(agent_count):
    with open(local_path+f'/models/{filename}/listener_{i}.pkl', 'rb') as f:
        a = cloudpickle.load(f)
        listener_agents.append(a)
    with open(local_path+f'/models/{filename}/speaker_{i}.pkl', 'rb') as f:
        a = cloudpickle.load(f)
        speaker_agents.append(a)

l0 = listener_agents[0]

l0.apply_fn(l0.params, jnp.array([0]))  # This will return a distrax distribution which should be sampled to get spline parameters.

print("Done")
