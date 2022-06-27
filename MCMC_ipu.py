# IPU modules
from tensorflow.python.ipu import ipu_compiler, ipu_infeed_queue, loops, utils
from tensorflow.python.ipu.scopes import ipu_scope, ipu_shard
from tensorflow.python import ipu

# General tensorflow libs
import tensorflow_probability as tfp 
import tensorflow as tf 
import time as time 
import numpy as np 

# Configure the IPU system
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()
# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()

first_layer_size = 40 
num_burnin_steps = 100 
num_results = 600 
num_leapfrog_steps = 1000 
input_file = "returns_and_features_for_mcmc.txt"
useful_features = 22 
num_skip_columns =2 
output_file = "output_samples.txt"

## Load data 
# Load data
raw_data = np.genfromtxt(input_file, skip_header=1,
                         delimiter="\t", dtype='float32')

# Pre-process data
observed_return_ = raw_data[:, num_skip_columns]
observed_features_ = raw_data[:, num_skip_columns+1:]
num_features = raw_data.shape[1] - num_skip_columns - 1
if useful_features < num_features:
    num_features = useful_features
    observed_features_ = observed_features_[:, :num_features]

# Model is an MLP with num_features input dims and layer sizes: first_layer_size, 1, 1
num_model_parameters = num_features * first_layer_size + \
    first_layer_size + first_layer_size + 3

# Print dataset parameters
print(" Number of data items {}\n"
      " Number of features per data item {}\n"
      " Number of model parameters {}\n"
      .format(raw_data.shape[0],
              num_features,
              num_model_parameters
              ))
# Import TensorFlow modules
tfd = tfp.distributions

# Supress warnings 
print("Supressed warnings")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize TensorFlow graph and session
tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)

# Build the neural network
def bdnn(x, p):
    nf = num_features
    nt = first_layer_size

    # Unpack model parameters
    w1 = tf.reshape(p[nt+1:nt+nf*nt+1], [nf, nt])
    w2 = tf.reshape(p[1:nt+1], [nt, 1])
    w3 = p[0]
    b1 = p[nt+nf*nt+3:]
    b2 = tf.expand_dims(p[nt+nf*nt+2], 0)
    b3 = p[nt+nf*nt+1]

    # Build layers
    x = tf.tanh(tf.compat.v1.nn.xw_plus_b(x, w1, b1))
    x = tf.compat.v1.nn.xw_plus_b(x, w2, b2)
    x = x * w3 + b3
    return tf.squeeze(x)


# Model posterior log probability
def model_log_prob(ret, feat, p):
    # Parameters of distributions
    prior_scale = 200
    studentT_scale = 100

    # Features normalization
    def normalize_features(f):
        return 0.001 * f

    # Prior probability distributions on model parameters
    rv_p = tfd.Independent(tfd.Normal(loc=0. * tf.ones(shape=[num_model_parameters], dtype=tf.float32),
                                      scale=prior_scale * tf.ones(shape=[num_model_parameters], dtype=tf.float32)),
                           reinterpreted_batch_ndims=1)

    # Likelihood
    alpha_bp_estimate = bdnn(normalize_features(feat), p)
    rv_observed = tfd.StudentT(
        df=2.2, loc=alpha_bp_estimate, scale=studentT_scale)

    # Sum of logs
    return (rv_p.log_prob(p) +
            tf.reduce_sum(rv_observed.log_prob(ret)))

with ipu_scope('/device:IPU:0'): 

    print("variable casting")
    ## Data items 
    observed_return = tf.cast(observed_return_, 'float32')
    observed_features= tf.cast(observed_features_, 'float32')

    print("initailise chain state")
    # Intial chain state 
    initial_chain_state = [ 0.0 * tf.ones(shape=[num_model_parameters], dtype=tf.float32)]

    print("initilaise bijectors")
    # Bijectors
    unconstraining_bijectors = [tfp.bijectors.Identity()]

    print("initiliase the step size")
    # Initialize the step_size
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        step_size = tf.compat.v1.get_variable(
        name='step_size',
        initializer=tf.constant(.01, dtype=tf.float32),
        trainable=False,
        use_resource=True)
 
    # Target log prob function 
    print("reached target log prob function") 
    
    def hmc_graph(): 
    
        def target_log_prob_fn(*args):
            return model_log_prob(observed_return, observed_features, *args)

        print("Reach hamilonian monte carlo model")

        # Hamiltonian Monte Carlo Model 
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
                        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=target_log_prob_fn,
                        num_leapfrog_steps=num_leapfrog_steps,
                        step_size=step_size,
                        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                        target_rate=0.2,
                        num_adaptation_steps=num_burnin_steps,
                        decrement_multiplier=0.1),
                        state_gradients_are_stopped=False),
                        bijector=unconstraining_bijectors)

        print("graph to sample from the chain stage")
        # Graph to sample from the chain 
        
        return  tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_chain_state,
                kernel=hmc_kernel)
    
    [p], kernel_results = ipu_compiler.compile(hmc_graph, [])

# Configure the IPU 
config = utils.create_ipu_config()
# Create 1 TF device with 1 IPU per device 
config = utils.auto_select_ipus(config,[1]) #1 means there is only 1 IPU
utils.configure_ipu_system(config)
utils.move_variable_initialization_to_cpu()


# initialize variables 
print("run session")
init_g = tf.global_variables_initializer()
sess.run(init_g)

# Warm up 

print("\nWarming up")
sess.run([p, kernel_results])
print("Done\n")

# Sample 

print("Sampling..")
start_time = time.time()
[samples, kernel_results_] = sess.run([p, kernel_results])
end_time = time.time()
print("Done\n")

# Write samples to file 
np.savetxt(output_file, samples, delimiter='\t')
print("Written {} samples to {}".format(samples.shape[0], output_file))

# print run time 
print("Completed in {0:.2f} seconds\n".format(end_time - start_time))
