import cloudpickle
import pathlib
import sys
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
import orbax.checkpoint
import jax
import jax.numpy as jnp

class TrainState(train_state.TrainState):
    key: jax.Array

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
    def __init__(self, speaker_agent, listener_agent, speaker_action_transform_name="splines", image_dim=28) -> None:
        self.speaker = speaker_agent
        self.listener = listener_agent
        self.speaker_action_transform = get_speaker_action_transform(speaker_action_transform_name, image_dim)
    
    def speak(self, obs): # This will return a distrax distribution which should be sampled to get spline parameters.
        return self.speaker.apply_fn(self.speaker.params, obs)
    
    def listen(self, obs): # This will return a categorical distribution which should be sampled to get action indices.
        return self.listener.apply_fn(self.listener.params, obs)
    
    def interpret_pictogram(self, key, obs, speaker_action_dim=21, n_samples=3): # obs is an image
        # P(r|s) = P(s|r)P(r)
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        #### Find the referent for which P(r|s) is the highest

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 (should be unused!)
        signal_param_samples = signal_distribution.sample(seed=key, sample_shape=(speaker_action_dim, n_samples))[0]    # This is shaped (n_samples, 1, 12)
        
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

    def create_pictogram(self, key, obs, speaker_action_dim=21, n_samples=3, n_search=4): # obs is an integer between 0 and num_referents-1
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        key, numer_key, denom_key = jax.random.split(key, 3)

        ######### Search for the signal with the highest P(r|s)

        ###### Calculate denominator

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 (should be unused!)
        signal_param_samples = signal_distribution.sample(seed=denom_key, sample_shape=(speaker_action_dim, n_samples))[0]    # This is shaped (n_samples, 1, 12)
        
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
        signal_param_samples = signal_distribution.sample(seed=numer_key, sample_shape=(speaker_action_dim, n_search))[0]    # This is shaped (n_search, 1, 12)
        
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
    

def test():
    # filename = "agents-300e-73c6"   # For MNIST
    # filename = "agents-300e-8e97"   # For cifar10
    filename = "agents-cifar10-50e-3000dp-a165"
    agent_indices = (0, 1) # list(range(2))
    listener_agents = []
    speaker_agents = []
    superagents = []

    key = jax.random.PRNGKey(0)

    for i in agent_indices:
        ### Load listener agent
        listener_network = ActorCriticListenerConv(action_dim=10, image_dim=32, config={"LISTENER_DROPOUT": 0.0})
        init_x = jnp.zeros(
            (32**2,)
        )
        network_params = listener_network.init({'params': key, 'dropout': key, 'noise': key}, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=listener_network.apply,
            params=network_params,
            key=key,
            tx=tx,
        )

        empty_checkpoint = {'model': train_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(local_path+f'/models/{filename}/listener_{i}.agent', item=empty_checkpoint)
        train_state = raw_restored['model']
        listener_agents.append(train_state)

        ### Load speaker agent
        speaker_network = ActorCriticSpeakerSplines(latent_dim=128, num_classes=10+1, action_dim=21, config={"SPEAKER_STDDEV": 0.7, "SPEAKER_STDDEV2": 0.4, "SPEAKER_SQUISH": 0.4})
        init_x = jnp.zeros(
            (1,),
            dtype=jnp.int32
        )
        network_params = speaker_network.init({'params': key, 'dropout': key, 'noise': key}, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=speaker_network.apply,
            params=network_params,
            key=key,
            tx=tx,
        )

        empty_checkpoint = {'model': train_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(local_path+f'/models/{filename}/speaker_{i}.agent', item=empty_checkpoint)
        train_state = raw_restored['model']

        speaker_agents.append(train_state)
        
        ### Make superagent
        superagents.append(Superagent(speaker_agents[-1], listener_agents[-1], "splines_weight", 32))


    # key = jax.random.PRNGKey(0)
    # random_speak = get_speaker_action_transform("splines", 28)(o[0].sample(seed=key))
    # pi = s0.interpret_pictogram(key, random_speak, n_samples=500)
    # print(pi.sample(seed=key))
    # print("Done")
    
    agent0 = superagents[0]
    agent1 = superagents[1]
    num_iters = 5
    image_paths = []
    pictograms = []

    for i in range(num_iters):
        key, key_i = jax.random.split(key)
        sp_key, ls_key = jax.random.split(key_i, 2)
        pictogram = agent0.create_pictogram(sp_key, jnp.array([i % 10]), n_samples=500, n_search=50)
        pi = agent1.interpret_pictogram(ls_key, pictogram, n_samples=500)
        reading = pi.sample(seed=key_i)

        image_paths.append(local_path+f'/scratch/scratch_images/pic_{i}_sign_{i % 10}_read_{reading}.png')
        pictograms.append(pictogram)

    for path, pic in zip(image_paths, pictograms):
        plt.imsave(path, pic, cmap='gray')

if __name__ == "__main__":
    test()

