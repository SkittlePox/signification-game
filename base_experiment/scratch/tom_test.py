import cloudpickle
import pathlib
import sys
import jax.numpy as jnp

local_path = str(pathlib.Path().resolve())+"/base_experiment"
sys.path.append(local_path)
local_path = str(pathlib.Path().resolve().parent)
sys.path.append(local_path)
from agents import *


class Superagent:
    def __init__(self, speaker_agent, listener_agent) -> None:
        self.speaker = speaker_agent
        self.listener = listener_agent
    
    def speak(self, obs): # This will return a distrax distribution which should be sampled to get spline parameters.
        return self.speaker.apply_fn(self.speaker.params, obs)
    
    def listen(self, obs): # This will return a categorical distribution which should be sampled to get action indices.
        return self.listener.apply_fn(self.listener.params, obs)
    
    def interpret_pictogram(self, obs):
        pass

    def create_pictogram(self, obs):
        pass
    


filename = "agents-300e-73c6"
agent_indices = {3, 4} # list(range(2))
listener_agents = []
speaker_agents = []
superagents = []

for i in agent_indices:
    with open(local_path+f'/models/{filename}/listener_{i}.pkl', 'rb') as f:
        a = cloudpickle.load(f)
        listener_agents.append(a)
    with open(local_path+f'/models/{filename}/speaker_{i}.pkl', 'rb') as f:
        a = cloudpickle.load(f)
        speaker_agents.append(a)
    superagents.append(Superagent(speaker_agents[-1], listener_agents[-1]))


s0 = superagents[0]
o = s0.listen(jnp.ones((28, 28)))
print(o)
o = s0.speak(jnp.array([0]))
print(o)

print("Done")
