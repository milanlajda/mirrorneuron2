from pybrain2.structure.networks import FeedForwardNetwork
from pybrain2.structure.modules import LinearLayer, SigmoidLayer
from pybrain2.structure.connections import FullConnection

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


