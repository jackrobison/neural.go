package neural

import (
	"math"
)

const bias = -1.0

// represents a vertex in the neuron graph
type Neuron struct {
	Inputs     []*Synapse // edges in
	Outputs    []*Synapse // edges out
	Activation float64    // node activation
	Sum        float64    // the sum of the input signals and weights (before being normalized by sigma)
	Deltas     []float64  // error deltas for the batch (used to calculate batch gradients
	//						 and ultimately the mean error gradient
}

func NewNeuron() *Neuron {
	n := &Neuron{}
	n.Activation = 0.0
	return n
}

func (n *Neuron) Sigmoid(x float64) float64 {
	result := math.Tanh(x)
	return Round(result)
}

func (n *Neuron) DSigmoid(x float64) float64 {
	result := 1.0 - (n.Sigmoid(x) * n.Sigmoid(x))
	return Round(result)
}

func (n *Neuron) Fire() {
	n.Sum = bias
	for _, synapseIn := range n.Inputs {
		synapseIn.Fire()
	}
	n.Activation = n.Sigmoid(n.Sum)
}

func (n *Neuron) Reset() {
	n.Activation = 0.0
	n.Sum = 0.0
}
