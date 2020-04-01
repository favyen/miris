package refine

import (
	"../miris"
	"../predicate"

	"fmt"
	"math"
	"strconv"
	"sort"
)

func GetCoarseIntermediate(freq int, k int, track []miris.Detection) []miris.Detection {
	start := -1
	end := -1
	var coarse []miris.Detection
	for i, detection := range track {
		if detection.FrameIdx % freq != k {
			continue
		}
		coarse = append(coarse, detection)
		if start == -1 {
			start = i
		}
		end = i
	}
	if start == -1 || end == -1 {
		return nil
	}
	// add interpolated frames
	var out []miris.Detection
	out = append(out, track[0:start]...)
	out = append(out, coarse[0])
	for _, detection := range coarse[1:] {
		last := out[len(out)-1]
		for frameIdx := last.FrameIdx+freq; frameIdx < detection.FrameIdx; frameIdx += freq {
			out = append(out, miris.Interpolate(last, detection, frameIdx))
		}
		out = append(out, detection)
	}
	out = append(out, track[end:]...)
	return out
}

type AccelRefiner struct {
	freq int
	predFunc predicate.Predicate
	accelThreshold float64
}

func MakeAccelRefiner(freq int, trainTracks [][]miris.Detection, predFunc predicate.Predicate, cfg map[string]string) Refiner {
	r := &AccelRefiner{
		freq: freq,
		predFunc: predFunc,
	}
	if cfg["threshold"] != "" {
		var err error
		r.accelThreshold, err = strconv.ParseFloat(cfg["threshold"], 64)
		if err != nil {
			panic(err)
		}
	}
	return r
}

func init() {
	InterpRefiners["accel"] = MakeAccelRefiner
}

func (r *AccelRefiner) insertDetection(coarse []miris.Detection, original []miris.Detection, frameIdx int) []miris.Detection {
	// find index in coarse where we need to insert a detection
	var insertIdx int = 0
	for i := 0; i < len(coarse); i++ {
		if coarse[i].FrameIdx < frameIdx {
			insertIdx = i+1
		}
	}
	if insertIdx == 0 || insertIdx == len(coarse) {
		fmt.Errorf("insertDetection got out of bounds frameIdx")
	}

	// if frameIdx appears in original, our job is easy
	for _, detection := range original {
		if detection.FrameIdx != frameIdx {
			continue
		}
		return append(coarse[0:insertIdx], append([]miris.Detection{detection}, coarse[insertIdx:]...)...)
	}

	// otherwise let's interpolate
	detection := miris.Interpolate(coarse[insertIdx-1], coarse[insertIdx], frameIdx)
	return append(coarse[0:insertIdx], append([]miris.Detection{detection}, coarse[insertIdx:]...)...)
}

// Returns frame indices and accel value along the segment with largest acceleration.
func (r *AccelRefiner) refineOnce(track []miris.Detection) (int, int, float64) {
	var bestAccel float64
	var bestIdx int = -1
	for i := 1; i < len(track)-1; i++ {
		if track[i].FrameIdx - track[i-1].FrameIdx <= 1 && track[i+1].FrameIdx - track[i].FrameIdx <= 1 {
			continue
		}
		vector1 := track[i].Bounds().Center().Sub(track[i-1].Bounds().Center())
		vector2 := track[i+1].Bounds().Center().Sub(track[i].Bounds().Center())
		accel := vector2.Sub(vector1).Magnitude()
		if bestIdx == -1 || accel > bestAccel {
			bestAccel = accel
			bestIdx = i
		}
	}
	if bestIdx == -1 {
		return -1, -1, -1
	}
	f1 := -1
	f2 := -1
	if track[bestIdx].FrameIdx - track[bestIdx-1].FrameIdx > 1 {
		f1 = (track[bestIdx-1].FrameIdx + track[bestIdx].FrameIdx) / 2
	}
	if track[bestIdx+1].FrameIdx - track[bestIdx].FrameIdx > 1 {
		f2 = (track[bestIdx].FrameIdx + track[bestIdx+1].FrameIdx) / 2
	}
	return f1, f2, bestAccel
}

func (r *AccelRefiner) Plan(valTracks [][]miris.Detection, bound float64) map[string]string {
	// for each coarse track, find the accelThreshold needed to get the predicate correct
	var precisionSamples, recallSamples []float64
	for _, track := range valTracks {
		label := r.predFunc([][]miris.Detection{track})
		for k := 0; k < r.freq; k++ {
			coarse := GetCoarseIntermediate(r.freq, k, track)
			if !label && r.predFunc([][]miris.Detection{coarse}) == label {
				// doesn't affect precision/recall
				continue
			} else if len(coarse) < 3 {
				if label {
					recallSamples = append(recallSamples, 0)
				}
				continue
			}
			var minAccelUsed float64 = 9999
			for r.predFunc([][]miris.Detection{coarse}) != label {
				f1, f2, accel := r.refineOnce(coarse)
				if accel < minAccelUsed {
					minAccelUsed = accel
				}
				if f1 != -1 {
					coarse = r.insertDetection(coarse, track, f1)
				}
				if f2 != -1 {
					coarse = r.insertDetection(coarse, track, f2)
				}
			}
			if label {
				recallSamples = append(recallSamples, minAccelUsed)
				precisionSamples = append(precisionSamples, 9999)
			} else {
				precisionSamples = append(precisionSamples, minAccelUsed)
			}
		}
	}
	sort.Float64s(precisionSamples)
	sort.Float64s(recallSamples)
	t1 := precisionSamples[int((1-bound)*float64(len(precisionSamples)))]
	t2 := recallSamples[int((1-bound)*float64(len(recallSamples)))]
	r.accelThreshold = math.Min(t1, t2)
	return map[string]string{
		"threshold": fmt.Sprintf("%v", r.accelThreshold),
	}
}

// TODO: instead of refineOnce, we should refine every step where accel exceeds threshold
func (r *AccelRefiner) Step(tracks [][]miris.Detection, seen []int) ([]int, []int) {
	needed := make(map[int]bool)
	var refined []int
	for i, track := range tracks {
		if len(track) < 3 {
			continue
		}
		track = miris.DensifyAt(track, seen)
		f1, f2, accel := r.refineOnce(track)
		if accel < r.accelThreshold {
			continue
		}
		if f1 != -1 {
			needed[f1] = true
		}
		if f2 != -1 {
			needed[f2] = true
		}
		refined = append(refined, i)
	}
	var neededList []int
	for frameIdx := range needed {
		neededList = append(neededList, frameIdx)
	}
	return neededList, refined
}

func (r *AccelRefiner) Close() {}
