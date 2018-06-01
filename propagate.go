package neural

func (f *FeedForwardNetwork) Propagate(inputs []float64) {
	f.Input(inputs)
	for i := 1; i < len(f.Layers); i++ {
		f.Layers[i].Fire()
	}
}
