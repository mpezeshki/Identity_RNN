import theano
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant, Identity
from blocks.algorithms import GradientDescent
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph
from utils import learning_algorithm
from datasets import MNIST

floatX = theano.config.floatX


n_epochs = 30
x_dim = 1
h_dim = 100
o_dim = 10
batch_size = 50

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
y = tensor.tensor3('y', dtype='int32')

x_to_h1 = Linear(name='x_to_h1',
                 input_dim=x_dim,
                 output_dim=h_dim)
pre_rnn = x_to_h1.apply(x)
rnn = SimpleRecurrent(activation=Rectifier(),
                      dim=h_dim, name="rnn")
h1 = rnn.apply(pre_rnn)
h1_to_o = Linear(name='h1_to_o',
                 input_dim=h_dim,
                 output_dim=o_dim)
pre_softmax = h1_to_o.apply(h1)
softmax = Softmax()
shape = pre_softmax.shape
softmax_out = softmax.apply(pre_softmax.reshape((-1, o_dim)))
softmax_out = softmax_out.reshape(shape)
softmax_out.name = 'softmax_out'

# comparing only last time-step
cost = CategoricalCrossEntropy().apply(y[-1, :, 0], softmax_out[-1])
cost.name = 'CrossEntropy'
error_rate = MisclassificationRate().apply(y[-1, :, 0], softmax_out[-1])
error_rate.name = 'error_rate'

# Initialization
for brick in (x_to_h1, h1_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()
rnn.weights_init = Identity()
rnn.biases_init = Constant(0)
rnn.initialize()

print 'Bulding training process...'
algorithm = GradientDescent(
    cost=cost,
    parameters=ComputationGraph(cost).parameters,
    step_rule=learning_algorithm(learning_rate=1e-6, momentum=0.0,
                                 clipping_threshold=1.0, algorithm='adam'))

train_stream, valid_stream = MNIST(batch_size=batch_size)

monitor_train_cost = TrainingDataMonitoring([cost, error_rate],
                                            prefix="train",
                                            after_epoch=True)

monitor_valid_cost = DataStreamMonitoring([cost, error_rate],
                                          data_stream=valid_stream,
                                          prefix="train",
                                          after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=[monitor_train_cost,
                                 monitor_valid_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print 'Starting training ...'
main_loop.run()
