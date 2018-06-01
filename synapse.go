package neural

// represents an edge in the feed forward neuron graph
// *Neuron --Synapse--> *Neuron
type Synapse struct {
	NeuronIn         *Neuron // pointer to input neuron
	NeuronOut        *Neuron // pointer to output neuron
	Weight           float64 // weight of the connection
	Gradient         float64 // δ error / δ weight
	PreviousGradient float64
	Velocity         float64
}

func (s *Synapse) Fire() {
	s.NeuronOut.Sum += Round(s.Weight * s.NeuronIn.Activation)
}
