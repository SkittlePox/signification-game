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
    
    def interpret_pictogram(self, key, obs, speaker_action_dim=21, n_samples=3, num_classes=10, pr_weight=1.5): # obs is an image
        # P(r_i|s) p= P(s|r_i)P(r_i)P_R(r_i)
        # P_R(r_i) = 1 / int_S f_i(s)p(s) ds + epsilon
        #   or in logits l_i terms: 1 / int_S exp(l_i(s)) p(s) ds + epsilon
        # P(s|r_i) p= f_i(s) / sum f_j(s) for j != i

        ###### Find the referent for which P(r|s) is the highest

        #### Calculate P(s|r_i) for each possible referent
        # This is a simple operation

        listener_assessments = self.listener.apply_fn(self.listener.params, obs)[0] # This is a categorical distribution. Index 1 is values
        
        def calc_psr(logits, index):
            numerator = logits[index]
            denominator = jax.nn.logsumexp(logits.at[index].set(-jnp.inf), axis=0)  # Set index to neg inf to remove it from denominator sum
            return numerator - denominator

        vmap_calc_psr = jax.vmap(calc_psr, in_axes=(None, 0))
        log_psrs = vmap_calc_psr(listener_assessments.logits[0], jnp.arange(num_classes))

        #### Calculate P_R(r_i) for each referent

        # Sample signal space n_samples times
        # This should be done using samples from all action generators. Can also be done using the extra action generator.
        # Below currently samples from generator 0, which is reserved for making observations of class 0. This is only okay because it's a scratch experiment. In the future should use instances from all generators!
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array(jnp.arange(num_classes), dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values.
        signal_param_samples = signal_distribution.sample(seed=key, sample_shape=(n_samples)).reshape((-1, speaker_action_dim))    # This is shaped (n_samples*num_classes, speaker_action_dim). The reshape is to merge samples from all generators together
        
        # Generate actual images
        signal_samples = self.speaker_action_transform(signal_param_samples)    # This is shaped (n_samples, 32, 32)

        listener_assessments = self.listener.apply_fn(self.listener.params, signal_samples)[0] # This is a categorical distribution. Index 1 is values
        # This has nearly everything we need. At this point I could take the logits and do the calculation

        # Sum exponentiated logits using exp(logsumexp) vertically?
        log_pRs = -(jax.nn.logsumexp(listener_assessments.logits, axis=0) - jnp.log(n_samples))    # These should technically be multiplied by p(s) before summing, but assuming random uniform dist I'm just dividing by n_samples
        print(jnp.exp(log_pRs))
        log_pRs *= pr_weight        # NOTE: This will definitely need to be tuned. Between 0.1 and 1.0 I'm guessing. Maybe need a sweep later.
        print(jnp.exp(log_pRs))

        #### Calculate P(r_i|s)

        log_prss = log_psrs + log_pRs - jnp.log(num_classes) # Assuming uniform random referent distribution means I can divide by num_classes. Using log rules
        log_prss -= jax.nn.logsumexp(log_prss)
        prss = jnp.exp(log_prss)
        print(prss)
        print("================")

        pictogram_pi = distrax.Categorical(probs=prss)

        return pictogram_pi
    
    def interpret_pictogram_old_simple(self, key, obs, speaker_action_dim=21, n_samples=3, num_classes=10): # obs is an image
        # P(r|s) = P(s|r)P(r)
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        #### Find the referent for which P(r|s) is the highest

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        # This should be done using samples from all action generators. Can also be done using the extra action generator.
        # Below currently samples from generator 0, which is reserved for making observations of class 0. This is only okay because it's a scratch experiment. In the future should use instances from all generators!
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 for now (should be unused in future!)
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
    
    def create_pictogram(self, key, obs, speaker_action_dim=21, n_samples=3, n_search=4, num_classes=10): # obs is an integer between 0 and num_referents-1
        # P(s|r_i) p= f_i(s) / sum f_j(s) for j != i       here, f_i(s) represents unnormalized probabilites.
        # In terms of logits exp(l_i(s)) = f_i(s)
        # P(s|r_i) p= exp( l_i(s) - log sum exp(l_j(s)) for j != i )
        
        key, numer_key, denom_key = jax.random.split(key, 3)

        ######### Search for the signal with the highest P(s|r)

        ###### Generate candidate stimuli

        # Sample the generator n_search times.
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([obs], dtype=jnp.int32))[0]    # This is a distrax distribution
        signal_param_samples = signal_distribution.sample(seed=denom_key, sample_shape=(speaker_action_dim, n_search))[0]    # This is shaped (n_search, 1, speaker_action_dim)
        
        # Generate actual images
        signal_samples = self.speaker_action_transform(signal_param_samples)    # This is shaped (n_search, 32, 32)

        # Get listener assessments l_i for each signal s
        listener_assessments = self.listener.apply_fn(self.listener.params, signal_samples)[0] # This is a categorical distribution. Index 1 is values
        # This has nearly everything we need. At this point I could take the logits, the probs, or the logprobs and do the calculation

        ###### Calculate P(s|r) using l_i values
        
        # Retrieve listener logits for each of the n_search samples
        listener_logits = listener_assessments.logits # This is shaped (n_search, 10)
        
        # Numerators are l_i(s)
        numerators = listener_logits[:, obs]

        # Set the logits for obs to -jnp.inf so they don't show up in the denominator calculation
        denominators = jax.nn.logsumexp(listener_logits.at[:, obs].set(-jnp.inf), axis=1)

        psr = jnp.exp(numerators-denominators)

        ###### Select signal with the highest P(s|r)
        
        maxindex = jnp.argmax(psr)
        return signal_samples[maxindex]
        

    def create_pictogram_old_simple(self, key, obs, speaker_action_dim=21, n_samples=3, n_search=4): # obs is an integer between 0 and num_referents-1
        # P(s|r) p= exp(U(s:r))
        # U(s:r) = log(P_lit(r|s))
        # P_lit(r|s) p= f_r(s) / int_S f_r(s') ds'  P(r)

        key, numer_key, denom_key = jax.random.split(key, 3)

        ######### Search for the signal with the highest P(r|s)

        ###### Calculate denominator

        # Sample signal space n times (how is this done? By using existing actions or new actions for signals that haven't been used before?) This will be used for denominator of P_lit        
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 for now (should be unused in future!)
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
        signal_distribution = self.speaker.apply_fn(self.speaker.params, jnp.array([0], dtype=jnp.int32))[0]    # This is a distrax distribution. Index 1 is values. Using speaker index 0 for now (should be unused in future!)
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
    filename = "agents-cifar10-2000e-5000dp-b770"
    agent_indices = (0, 1) # list(range(2))
    listener_agents = []
    speaker_agents = []
    superagents = []

    key = jax.random.PRNGKey(1)

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
    num_iters = 20
    image_paths = []
    pictograms = []

    for i in range(num_iters):
        key, key_i = jax.random.split(key)
        sp_key, ls_key = jax.random.split(key_i, 2)
        pictogram = agent0.create_pictogram(sp_key, jnp.array([i % 10]), n_search=100)
        pi = agent1.interpret_pictogram(ls_key, pictogram, n_samples=10)
        reading = pi.sample(seed=key_i)

        image_paths.append(local_path+f'/scratch/scratch_images/pic_{i}_sign_{i % 10}_read_{reading}.png')
        pictograms.append(pictogram)

    for path, pic in zip(image_paths, pictograms):
        plt.imsave(path, pic, cmap='gray')

if __name__ == "__main__":
    test()
