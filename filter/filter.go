package filter

import (
	"../miris"
)

const Threads int = 12

type FilterFunc func(freq int, trainTracks [][]miris.Detection, labels []bool, cfg map[string]string) Filter

type Filter interface {
	Predict(valTracks [][]miris.Detection) []float64
	Close()
}

var FilterMap = make(map[string]FilterFunc)
