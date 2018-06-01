package neural

type TrainingSet struct {
	Inputs  [][]float64
	Targets [][]float64
}

var XorTrainingSet = TrainingSet{
	[][]float64{
		{0.0, 0.0},
		{1.0, 1.0},
		{1.0, 0.0},
		{0.0, 1.0},
	},
	[][]float64{
		{0.0},
		{0.0},
		{1.0},
		{1.0},
	},
}
