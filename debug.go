package neural

import (
	"fmt"
)

func (n *FeedForwardNetwork) PrintWeights() {
	for i := 0; i < len(n.Layers); i++ {
		for j := range n.Layers[i].Neurons {
			for k := range n.Layers[i].Neurons[j].Outputs {
				fmt.Println("Layer ", i+1, " node ", j+1, " --> ", k+1, ": ", n.Layers[i].Neurons[j].Outputs[k].Weight)
			}
		}
	}
}

func (n *FeedForwardNetwork) RunTest(trainingSet *TrainingSet) {
	for x, i := range trainingSet.Inputs {
		n.Reset()
		n.Propagate(i)
		t := XorTrainingSet.Targets[x]
		for y, outputNeuron := range n.Layers[len(n.Layers)-1].Neurons {
			fmt.Println("input:", i, ", target:", t[y], ", output:", outputNeuron.Activation)

		}
	}
}
