package neural

import (
	"errors"
)

type FeedForwardLayer struct {
	Neurons []*Neuron
}

func (l *FeedForwardLayer) Reset() {
	for _, n := range l.Neurons {
		n.Reset()
	}
}

func (l *FeedForwardLayer) Fire() {
	for _, n := range l.Neurons {
		n.Fire()
	}
}

func (l *FeedForwardLayer) NormalizedInput(vals []float64) error {
	if len(vals) != len(l.Neurons) {
		return errors.New("invalid number of inputs")
	}
	for i, v := range vals {
		l.Neurons[i].Activation = l.Neurons[i].Sigmoid(v)
	}
	return nil
}
