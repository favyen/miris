package planner

import (
	"../gnn"
	"../miris"
	"../predicate"

	"log"
	"sort"
)

const QThreads int = 8

func GetQSamples(ppCfg miris.PreprocessConfig, segment miris.Segment, freq int, bound float64, modelPath string, predFunc predicate.Predicate) []float64 {
	log.Printf("[plan-q] begin %s", segment.FramePath)
	model := gnn.NewGNN(modelPath, segment.TrackPath, segment.FramePath, ppCfg.FrameScale)
	defer model.Close()
	detections := miris.ReadDetections(segment.TrackPath)
	tracks := miris.GetTracks(detections)
	goodTracks := make(map[int]bool)
	for _, track := range tracks {
		if !predFunc([][]miris.Detection{track}) {
			continue
		}
		goodTracks[track[0].TrackID] = true
	}

	// largest Q threshold needed to correctly get each track
	trackMaxQs := make(map[int]float64)

	for frameIdx := 0; frameIdx < len(detections)-freq; frameIdx += freq {
		log.Printf("[plan-q] %s ... %d/%d", segment.FramePath, frameIdx, len(detections))
		idx1 := frameIdx
		idx2 := frameIdx+freq
		mat := model.Infer(idx1, idx2)

		// for each left detection:
		// (1) get max probability over outgoing edges
		// (2) get q_frac as fraction of correct track probability
		for i, leftDet := range detections[idx1] {
			if !goodTracks[leftDet.TrackID] {
				continue
			}
			var maxProb float64 = mat[i][len(detections[idx2])]
			var matchProb float64 = -1
			for j, rightDet := range detections[idx2] {
				prob := mat[i][j]
				if prob > maxProb {
					maxProb = prob
				}
				if leftDet.TrackID == rightDet.TrackID {
					matchProb = prob
				}
			}
			if maxProb < 0.01 {
				continue
			}
			var curQ float64
			if matchProb == -1 {
				curQ = mat[i][len(detections[idx2])] / maxProb
			} else {
				curQ = matchProb / maxProb
			}
			oldQ, ok := trackMaxQs[leftDet.TrackID]
			if !ok || curQ < oldQ {
				trackMaxQs[leftDet.TrackID] = curQ
			}
		}
	}

	var samples []float64
	for _, q := range trackMaxQs {
		samples = append(samples, q)
	}
	return samples
}

type qjob struct {
	freq int
	segment miris.Segment
}

func PlanQ(maxFreq int, bound float64, ppCfg miris.PreprocessConfig, modelCfg miris.ModelConfig) map[int]float64 {
	predFunc := predicate.GetPredicate(ppCfg.Predicate)

	jobch := make(chan qjob)
	donech := make(chan map[int][]float64)

	for i := 0; i < QThreads; i++ {
		go func() {
			m := make(map[int][]float64)
			for job := range jobch {
				samples := GetQSamples(ppCfg, job.segment, job.freq, bound, modelCfg.GetGNN(job.freq).ModelPath, predFunc)
				m[job.freq] = append(m[job.freq], samples...)
			}
			donech <- m
		}()
	}

	for freq := 2; freq <= maxFreq; freq *= 2 {
		for _, segment := range ppCfg.ValSegments {
			jobch <- qjob{freq, segment}
		}
	}
	close(jobch)
	allSamples := make(map[int][]float64)
	for i := 0; i < QThreads; i++ {
		m := <- donech
		for freq, samples := range m {
			allSamples[freq] = append(allSamples[freq], samples...)
		}
	}

	freqToQ := map[int]float64{1: 1}
	for freq, samples := range allSamples {
		sort.Float64s(samples)
		q := samples[int((1-bound)*float64(len(samples)))]
		freqToQ[freq] = q
		log.Printf("[plan-q] compute q=%v at freq=%d", q, freq)
	}
	return freqToQ
}

func PlanQAt(freq int, bound float64, modelPath string, predFunc predicate.Predicate, ppCfg miris.PreprocessConfig) float64 {
	var samples []float64
	for _, segment := range ppCfg.ValSegments {
		curSamples := GetQSamples(segment, freq, bound, modelPath, predFunc)
		samples = append(samples, curSamples...)
	}

	sort.Float64s(samples)
	q := samples[int((1-bound)*float64(len(samples)))]
	log.Printf("[plan-q] compute q=%v at freq=%d", q, freq)
	return q
}
