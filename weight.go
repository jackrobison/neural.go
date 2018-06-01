package neural

import (
	"crypto/rand"
	"math/big"
)

func GetRandomWeight() float64 {
	n, _ := rand.Int(rand.Reader, big.NewInt(1000000))
	bf, _ := new(big.Float).SetInt(n).Float64()
	return Round(float64(bf) / float64(1000000))
}
