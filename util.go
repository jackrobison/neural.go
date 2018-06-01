package neural

import (
	"math"
)

const sigfigs = 1.0E10

func Round(x float64) float64 {
	sign := 1.0
	if x*-1 > 0 {
		sign = -1.0
	}
	t := int64(math.Abs(x) * sigfigs)
	result := float64(t) / sigfigs * sign
	return result
}
