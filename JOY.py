from pybrain2.structure.networks  import FeedForwardNetwork
from pybrain2.structure.modules   import LinearLayer, SigmoidLayer
from pybrain2.structure.connections import FullConnection

from pybrain2.datasets            import ClassificationDataSet
from pybrain2.utilities           import percentError
from pybrain2.tools.shortcuts     import buildNetwork
from pybrain2.supervised.trainers import BackpropTrainer
from pybrain2.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


# Network #######################################################################
joynetwork = FeedForwardNetwork()
inputLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outputLayer = LinearLayer(1)

joynetwork.addInmutMoule(inputLayer)
joynetwork.addModule(hiddenLayer)
joynetwork.addOutputModule(outputLayer)

input_to_hidden = FullConnection(inputLayer, hiddenLayer)
hidden_to_output = FullConnection(hiddenLayer, outputLayer)

joynetwork.addConnection(input_to_hidden)
joynetwork.addConnection(hidden_to_output)

joynetwork.sortModules()

# Network #######################################################################


# Dataset - checknut co to vsetko znamena #######################################################

means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])

tstdata, trndata = alldata.splitWithProportion( 0.25 )