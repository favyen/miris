package planner

import (
	"../filter"
	"../miris"

	"log"
	"sort"
)

func PlanFilter(context plannerContext) miris.FilterPlan {
	trainLabels := make([]bool, len(context.trainTracks))
	for i, track := range context.trainTracks {
		trainLabels[i] = context.predFunc([][]miris.Detection{track})
	}
	valLabels := make([]bool, len(context.valTracks))
	var valTrue, valFalse int
	for i, track := range context.valTracks {
		valLabels[i] = context.predFunc([][]miris.Detection{track})
		if valLabels[i] {
			valTrue++
		} else {
			valFalse++
		}
	}

	// get coarse validation tracks
	// don't create much more than 25K true and 25K false tracks
	coarsePerTrueTrack := 1+(50000/valTrue)
	coarsePerFalseTrack := 1+(50000/valFalse)
	log.Printf("[plan-filter] compute coarse tracks (max true=%d false=%d per track with %d val tracks)", coarsePerTrueTrack, coarsePerFalseTrack, len(context.valTracks))
	var coarseTracks [][]miris.Detection
	var coarseLabels []bool
	for i, track := range context.valTracks {
		var n int
		if valLabels[i] {
			n = coarsePerTrueTrack
		} else {
			n = coarsePerFalseTrack
		}
		for j, coarse := range miris.GetAllCoarse(track, context.freq) {
			coarseTracks = append(coarseTracks, coarse)
			coarseLabels = append(coarseLabels, valLabels[i])
			if j >= n {
				break
			}
		}
	}

	// score the filters, need (bound) recall
	var bestName string
	var bestPrecision, bestThreshold float64
	for name, filterFunc := range filter.FilterMap {
		log.Printf("[plan-filter] trying out filter %s", name)
		curFilter := filterFunc(context.freq, context.trainTracks, trainLabels, context.modelCfg.GetFilterCfg(name, context.freq))
		precision, threshold := GetPrecisionAndThreshold(curFilter, coarseTracks, coarseLabels, context.bound)
		curFilter.Close()
		log.Printf("[plan-filter] ... %s: precision=%v at threshold=%v", name, precision, threshold)
		if precision > bestPrecision {
			bestName = name
			bestPrecision = precision
			bestThreshold = threshold
		}
	}
	log.Printf("[plan-filter] best filter is %s with precision=%v, threshold=%v", bestName, bestPrecision, bestThreshold)
	return miris.FilterPlan{
		Name: bestName,
		Threshold: bestThreshold,
	}
}

func GetPrecisionAndThreshold(curFilter filter.Filter, tracks [][]miris.Detection, labels []bool, bound float64) (float64, float64) {
	scores := curFilter.Predict(tracks)
	var trueScores, falseScores []float64
	for i := range tracks {
		if labels[i] {
			trueScores = append(trueScores, scores[i])
		} else {
			falseScores = append(falseScores, scores[i])
		}
	}

	sort.Float64s(trueScores)
	threshold := trueScores[int((1-bound)*float64(len(trueScores)))]
	var tp, fp int
	for _, score := range trueScores {
		if score >= threshold {
			tp++
		}
	}
	for _, score := range falseScores {
		if score >= threshold {
			fp++
		}
	}
	precision := float64(tp) / float64(tp+fp)
	return precision, threshold
}
