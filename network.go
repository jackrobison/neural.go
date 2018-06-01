package neural

import (
	"errors"
)

type FeedForwardNetwork struct {
	Layers   []*FeedForwardLayer
	Topology []int
}

func (f *FeedForwardNetwork) Reset() {
	for _, l := range f.Layers {
		l.Reset()
	}
}

func (f *FeedForwardNetwork) Input(inputs []float64) error {
	if len(f.Layers) < 1 {
		return errors.New("not initialized")
	}
	f.Layers[0].NormalizedInput(inputs)
	return nil
}

func (n *FeedForwardNetwork) AddInputLayer(inputCount int) error {
	if len(n.Layers) > 0 {
		return errors.New("input layer is already initialized")
	}
	l := &FeedForwardLayer{}
	for i := 0; i < inputCount; i++ {
		newNeuron := NewNeuron()
		l.Neurons = append(l.Neurons, newNeuron)
	}
	n.Layers = append(n.Layers, l)
	return nil
}

func (n *FeedForwardNetwork) AddLayer(size int) error {
	if len(n.Layers) < 1 {
		return errors.New("input layer is not initialized")
	}
	if size < 1 {
		return errors.New("invalid layer size")
	}
	l := &FeedForwardLayer{}
	for i := 0; i < size; i++ {
		newNeuron := NewNeuron()
		for j := range n.Layers[len(n.Layers)-1].Neurons {
			neuronIn := n.Layers[len(n.Layers)-1].Neurons[j]
			s := &Synapse{}
			s.NeuronIn = neuronIn
			s.NeuronOut = newNeuron
			s.Weight = GetRandomWeight()
			s.Gradient = 0.0

			// connect new neuron to neurons in previous layer
			newNeuron.Inputs = append(newNeuron.Inputs, s)

			// connect previous layer to new neuron
			neuronIn.Outputs = append(neuronIn.Outputs, s)

		}
		// now that the neuron has been initialized and connected,
		// add it to the pending layer
		l.Neurons = append(l.Neurons, newNeuron)

	}
	// add the finished layer to the network
	n.Layers = append(n.Layers, l)
	return nil
}

func (n *FeedForwardNetwork) Reinitialize() error {
	f := &FeedForwardNetwork{}
	f.Topology = n.Topology
	err := f.AddInputLayer(n.Topology[0])
	n.Layers[0] = f.Layers[0]
	if err != nil {
		return err
	}
	for i := 1; i < len(f.Topology); i++ {
		err = f.AddLayer(f.Topology[i])
		if err != nil {
			return err
		}
		n.Layers[i] = f.Layers[i]
	}
	return nil
}

func InitializeNetwork(inputCount int, neuronsInLayer ...int) (error, *FeedForwardNetwork) {
	f := &FeedForwardNetwork{}
	f.Topology = append(f.Topology, inputCount)
	err := f.AddInputLayer(inputCount)
	if err != nil {
		return err, f
	}
	for _, neuronCount := range neuronsInLayer {
		err = f.AddLayer(neuronCount)
		if err != nil {
			return err, f
		}
		f.Topology = append(f.Topology, neuronCount)
	}
	return nil, f
}
