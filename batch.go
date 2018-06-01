package neural

import (
	"errors"
)

func (n *FeedForwardNetwork) BatchRProp(inputSet, targetSet [][]float64, rateUp, rateDown float64) error {
	if len(n.Layers) < 2 {
		return errors.New("not initialized")
	}

	if len(inputSet) != len(targetSet) {
		return errors.New("amount of input samples does not match amount of target outputs")
	}
	for x := 0; x < len(inputSet); x++ {
		if len(inputSet[x]) != len(n.Layers[0].Neurons) {
			return errors.New("wrong number of inputs")
		}
		if len(targetSet[x]) != len(n.Layers[len(n.Layers)-1].Neurons) {
			return errors.New("wrong number of targets")
		}

		n.Reset()
		n.Propagate(inputSet[x])
		targets := targetSet[x]

		for i := len(n.Layers) - 1; i >= 0; i-- { // back-propagate the error
			layer := n.Layers[i]
			if i == len(n.Layers)-1 { // this is the output layer
				for j := range layer.Neurons {
					node := layer.Neurons[j]
					// E = actual - target
					// d[-1] = -E * sigma'(x)
					node.Deltas = append(node.Deltas, -1.0*(node.Activation-targets[j])*node.DSigmoid(node.Sum))
				}
			} else { // this is a hidden layer
				for j := range layer.Neurons {
					node := layer.Neurons[j]
					// d[i] = sigma'(x) * sum(w[i] * d[k])
					weightSum := 0.0
					for k := range node.Outputs {
						neuronOut := node.Outputs[k].NeuronOut
						weightSum += node.Outputs[k].Weight * neuronOut.Deltas[len(neuronOut.Deltas)-1]
					}
					node.Deltas = append(node.Deltas, node.DSigmoid(node.Sum)*weightSum)
					// gradient[l][i][j] = activation[l][i] * d[l][j]
					for k := range node.Outputs {
						o := node.Outputs[k]
						o.Gradient += node.Activation * o.NeuronOut.Deltas[len(o.NeuronOut.Deltas)-1]
					}
				}
			}
		}
	}

	// calculate the mean error gradient and update the node weights according to rprop
	for i := 1; i < len(n.Layers); i++ {
		for j := range n.Layers[i].Neurons {
			meanDelta := 0.0
			for d := range n.Layers[i].Neurons[j].Deltas {
				meanDelta += n.Layers[i].Neurons[j].Deltas[d]
			}
			meanDelta = meanDelta / float64(len(targetSet))
			n.Layers[i].Neurons[j].Deltas = []float64{}

			for k := range n.Layers[i].Neurons[j].Inputs {
				synapseIn := n.Layers[i].Neurons[j].Inputs[k]
				synapseInGradient := synapseIn.Gradient / float64(len(targetSet))
				synapseIn.Gradient = 0.0

				if synapseIn.Velocity == 0.0 {
					synapseIn.Velocity = 0.001
				}

				if synapseIn.PreviousGradient*synapseInGradient < 0 {
					synapseIn.Velocity = synapseIn.Velocity * rateDown
				} else {
					synapseIn.Velocity = synapseIn.Velocity * rateUp
				}

				if (synapseIn.Velocity * synapseInGradient) < 0 {
					synapseIn.Velocity = synapseIn.Velocity * -1
				}
				synapseIn.Weight += synapseIn.Velocity
				synapseIn.PreviousGradient = synapseInGradient
			}
		}
	}

	return nil
}
