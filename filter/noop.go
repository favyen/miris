package filter

import (
	"github.com/favyen/miris/miris"
)

func init() {
	FilterMap["noop"] = MakeNoopFilter
}

func MakeNoopFilter(freq int, tracks [][]miris.Detection, labels []bool, cfg map[string]string) Filter {
	return NoopFilter{}
}

type NoopFilter struct{}

func (noop NoopFilter) Predict(tracks [][]miris.Detection) []float64 {
	scores := make([]float64, len(tracks))
	return scores
}

func (noop NoopFilter) Close() {}
