package neural

import (
	"fmt"
	"math"
)

func (n *FeedForwardNetwork) rPropConverge(trainingSet *TrainingSet, maxRounds int, rateUp, rateDown float64) (error, int) {
	gradSum := 0.0
	prevGradSum := 0.0

	for c := 0; true; c++ {
		err := n.BatchRProp(trainingSet.Inputs, trainingSet.Targets, rateUp, rateDown)
		if err != nil {
			return err, c
		}
		// add up the absolute values of the error gradient, if we're at a minima this will approach zero
		gradSum = 0.0
		for i := range n.Layers {
			for j := range n.Layers[i].Neurons {
				for _, synapse := range n.Layers[i].Neurons[j].Inputs {
					gradSum += math.Abs(synapse.PreviousGradient)
				}
			}
		}
		if (math.Abs(gradSum-prevGradSum) <= 0.000001) && (c > 2) {
			return nil, c
		}
		if c >= maxRounds {
			return nil, c
		}

		prevGradSum = gradSum
	}
	fmt.Println("got to end")
	return nil, 0
}

func (n *FeedForwardNetwork) recursiveRPropConverge(trainingSet *TrainingSet, maxError, rateUp, rateDown float64, maxRounds int) (error, int) {
	/*
		This function recursively tries to rPropConverge until it finds a solution with a sufficiently low error
		when a minima is encountered, the network is re-initialized with random weights
	*/
	err, rounds := n.rPropConverge(trainingSet, maxRounds, rateUp, rateDown)
	if err != nil {
		return err, rounds
	}
	errorSum := 0.0
	for x, i := range trainingSet.Inputs {
		n.Reset()
		n.Propagate(i)
		t := trainingSet.Targets[x]
		for y, outputNeuron := range n.Layers[len(n.Layers)-1].Neurons {
			errorSum += math.Abs(t[y] - outputNeuron.Activation)
		}
	}
	if errorSum < maxError {
		fmt.Println("error:", errorSum)
		return nil, rounds
	}
	// randomize the network to try getting out of the minima
	n.Reinitialize()
	err, r := n.recursiveRPropConverge(trainingSet, maxError, rateUp, rateDown, maxRounds)
	return err, rounds + r
}

func (n *FeedForwardNetwork) RPropConverge(trainingSet *TrainingSet, maxError, rateUp, rateDown float64, maxRounds int) (error, int) {
	err, rounds := n.recursiveRPropConverge(trainingSet, maxError, rateUp, rateDown, maxRounds)
	if err != nil {
		return err, 0
	}
	fmt.Println("converged after ", rounds, "rounds")
	return nil, rounds
}
