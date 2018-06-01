package neural

import (
	"testing"
)

func TestInitializeNetwork(t *testing.T) {
	err, net := InitializeNetwork(2, 3, 1)
	if err != nil {
		t.Error("failed to initialize: ", err)
	}
	if len(net.Layers) != 3 {
		t.Error("failed to initialize")
	}
	if len(net.Layers[0].Neurons) != 2 {
		t.Error("layer has incorrect number of nodes")
	}
	if len(net.Layers[1].Neurons) != 3 {
		t.Error("layer has incorrect number of nodes")
	}
	if len(net.Layers[2].Neurons) != 1 {
		t.Error("layer has incorrect number of nodes")
	}
}

func TestConverge(t *testing.T) {
	err, net := InitializeNetwork(2, 3, 1)
	if err != nil {
		t.Error("failed to initialize: ", err)
	}
	err, _ = net.RPropConverge(&XorTrainingSet, 0.001, 1.3, 0.4, 1000)
	if err != nil {
		t.Error("failed to train: ", err)
	}
	net.RunTest(&XorTrainingSet)
	net.PrintWeights()
}
