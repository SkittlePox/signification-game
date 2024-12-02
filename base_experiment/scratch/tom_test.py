import cloudpickle
import pathlib
import sys
import jax.numpy as jnp
import distrax

DEBUGGER = "DEBUGGER=True" in sys.argv

local_path = str(pathlib.Path().resolve().parent)
sys.path.append(local_path)
local_path = str(pathlib.Path().resolve())+"/base_experiment"
sys.path.append(local_path)

if DEBUGGER:
    local_path = str(pathlib.Path().resolve())+"/base_experiment"
else:
    local_path = str(pathlib.Path().resolve().parent)

from agents import *
from utils import get_speaker_action_transform


class Superagent:
    def __init__(self, speaker_agent, listener_agent) -> None:
        self.speaker = speaker_agent
        self.listener = listener_agent
        self.speaker_action_transform = get_speaker_action_transform("splines", 28)
    
    def speak(self, obs): # This will return a distrax distribution which should be sampled to get spline parameters.
        return self.speaker.apply_fn(self.speaker.params, obs)
    
    def listen(self, obs): # This will return a categorical distribution which should be sampled to get action indices.
        return self.listener.apply_fn(self.listener.params, obs)
    
    def interpret_pictogram(self, key, obs, n_samples=3): # obs is an image
        # P(r|s) = P(s|r)P(r)
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        #### Find the referent for which P(r|s) is the highest

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 (should be unused!)
        signal_param_samples = signal_distribution.sample(seed=key, sample_shape=(12, n_samples))[0]    # This is shaped (n_samples, 1, 12)
        
        # Generate actual images
        signal_samples = self.speaker_action_transform(signal_param_samples)    # This is shaped (n_samples, 28, 28)

        listener_assessments = self.listener.apply_fn(self.listener.params, signal_samples)[0] # This is a categorical distribution. Index 1 is values
        # This has nearly everything we need. At this point I could take the logits, the probs, or the logprobs and do the calculation
        
        # Calculate denominators. Sum of probs
        listener_probs = listener_assessments.probs # This is shaped (n_samples, 10)
        # listener_logprobs = listener_assessments.log_prob(jnp.arange(10).reshape(10, 1))
        # listener_logits = listener_assessments.logits
        denominators = jnp.sum(listener_probs, axis=0)  # This is shaped (10,)
        numerators = self.listener.apply_fn(self.listener.params, obs)[0].probs[0] # This is shaped (10,)
        
        # p_lits = numerators / denominators    # Instead of dividing directly, for numerical stability I will use the log quotient rule
        p_lits = jnp.exp(jnp.log(numerators) - jnp.log(denominators))

        # Assuming uniform probability, so no need to multiply by P(r)
        # Could make a new distrax distribution, but this would involve importing distrax into simplified sig game, which I don't like
        pictogram_pi = distrax.Categorical(probs=p_lits)

        return pictogram_pi

    def create_pictogram(self, key, obs, n_samples=3, n_search=4): # obs is an integer between 0 and num_referents-1
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        key, numer_key, denom_key = jax.random.split(key, 3)

        ######### Search for the signal with the highest P(r|s)

        ###### Calculate denominator

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 (should be unused!)
        signal_param_samples = signal_distribution.sample(seed=denom_key, sample_shape=(12, n_samples))[0]    # This is shaped (n_samples, 1, 12)
        
        # Generate actual images
        signal_samples = self.speaker_action_transform(signal_param_samples)    # This is shaped (n_samples, 28, 28)

        listener_assessments = self.listener.apply_fn(self.listener.params, signal_samples)[0] # This is a categorical distribution. Index 1 is values
        # This has nearly everything we need. At this point I could take the logits, the probs, or the logprobs and do the calculation
        
        # Calculate denominators. Sum of probs
        listener_probs = listener_assessments.probs # This is shaped (n_samples, 10)
        # listener_logprobs = listener_assessments.log_prob(jnp.arange(10).reshape(10, 1))
        # listener_logits = listener_assessments.logits
        denominators = jnp.sum(listener_probs, axis=0)  # This is shaped (10,)

        ###### Search for signal and calculate numerator

        # (I think I'll want to include things in the search in the denominator too!)

        # Sample lots of possible signals.
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 (should be unused!)
        signal_param_samples = signal_distribution.sample(seed=numer_key, sample_shape=(12, n_search))[0]    # This is shaped (n_search, 1, 12)
        
        # Generate actual images
        signal_samples = self.speaker_action_transform(signal_param_samples)
        
        # Need to iterate over search space and find the highest numerator/denominator??
        numerators_all = self.listener.apply_fn(self.listener.params, signal_samples)[0].probs # This is shaped (n_search, 10)

        # Isolate obs referent index
        numerators = numerators_all[:, obs] # This is shape [n_search, 1]
        # Find the thing with the highest numerator for r? find the one with the highest r.

        maxindex = jnp.argmax(numerators)

        return signal_samples[maxindex]
        
        # p_lits = numerators / denominators    # Instead of dividing directly, for numerical stability I will use the log quotient rule
        p_lits = jnp.exp(jnp.log(numerators) - jnp.log(denominators))

        # Assuming uniform probability, so no need to multiply by P(r)
        # Could make a new distrax distribution, but this would involve importing distrax into simplified sig game, which I don't like
        pictogram_pi = distrax.Categorical(probs=p_lits)

        return pictogram_pi
    


filename = "agents-300e-73c6"
agent_indices = (3, 4) # list(range(2))
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


# s0 = superagents[0]
# o = s0.listen(jnp.ones((28, 28)))
# print(o)
# o = s0.speak(jnp.array([0]))
# print(o)


# key = jax.random.PRNGKey(0)
# random_speak = get_speaker_action_transform("splines", 28)(o[0].sample(seed=key))
# pi = s0.interpret_pictogram(key, random_speak, n_samples=500)
# print(pi.sample(seed=key))
# print("Done")

key = jax.random.PRNGKey(0)
sp_key, ls_key = jax.random.split(key, 2)

agent0 = superagents[0]
agent1 = superagents[1]
pictogram = agent0.create_pictogram(sp_key, jnp.array([5]), n_samples=1000, n_search=1000)
pi = agent1.interpret_pictogram(ls_key, pictogram, n_samples=1000)
print(pi.sample(seed=key))
print("Done")

import matplotlib.pyplot as plt
plt.imsave(local_path+'/scratch/pictogram.png', pictogram, cmap='gray')

