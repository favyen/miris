package planner

import (
	"github.com/favyen/miris/gnn"
	"github.com/favyen/miris/miris"
	"github.com/favyen/miris/predicate"

	"fmt"
	"log"
	"sort"
)

const QThreads int = 8

func GetQSamplesSegment(ppCfg miris.PreprocessConfig, segment miris.Segment, freq int, modelPath string, predFunc predicate.Predicate) []float64 {
	log.Printf("[plan-q] begin %s @ %d", segment.FramePath, freq)
	model := gnn.NewGNN(modelPath, segment.TrackPath, segment.FramePath, ppCfg.FrameScale)
	defer model.Close()
	detections := miris.ReadDetections(segment.TrackPath)
	tracks := miris.GetTracks(detections)

	// we will abuse the detection scores to fill in the gnn q values
	// that say whether the detection matches to the next one (or, for last
	// detection, to the terminal)
	scoredTracks := make(map[int][]miris.Detection)
	for _, track := range tracks {
		if !predFunc([][]miris.Detection{track}) {
			continue
		}
		scoredTracks[track[0].TrackID] = track
		for i := range track {
			track[i].Score = 1
		}
	}

	// return idx of detection in track with the specified frameIdx
	findTrackIdxByFrame := func(track []miris.Detection, frameIdx int) int {
		for i, detection := range track {
			if detection.FrameIdx == frameIdx {
				return i
			}
		}
		return -1
	}

	var frames [][2]int
	for frameIdx := 0; frameIdx < len(detections)-freq; frameIdx += freq {
		frames = append(frames, [2]int{frameIdx, frameIdx + freq})
	}
	mats := model.InferMany(frames, fmt.Sprintf("[plan-q] [%s @ %d]", segment.FramePath, freq))
	for counter, mat := range mats {
		idx1 := frames[counter][0]
		idx2 := frames[counter][1]

		// for each left detection:
		// (1) get max probability over outgoing edges
		// (2) get q_frac as fraction of correct track probability
		for i, leftDet := range detections[idx1] {
			if scoredTracks[leftDet.TrackID] == nil {
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
			var curQ float64
			if maxProb < 0.01 {
				curQ = 1
			} else if matchProb == -1 {
				curQ = mat[i][len(detections[idx2])] / maxProb
			} else {
				curQ = matchProb / maxProb
			}

			track := scoredTracks[leftDet.TrackID]
			idxInTrack := findTrackIdxByFrame(track, idx1)
			track[idxInTrack].Score = curQ
		}
	}

	// disconnects is set of disconnected indices in track
	// if i is disconnected, it means
	getLongestSegment := func(track []miris.Detection, disconnects map[int]bool) []miris.Detection {
		discList := []int{-1, len(track) - 1}
		for idx := range disconnects {
			discList = append(discList, idx)
		}
		sort.Ints(discList)
		var longestSegment []miris.Detection
		for i := 0; i < len(discList)-1; i++ {
			segment := track[discList[i]+1 : discList[i+1]+1]
			if len(segment) > len(longestSegment) {
				longestSegment = segment
			}
		}
		return longestSegment
	}

	var samples []float64
	for _, track := range scoredTracks {
		// iteratively disconnect the track at edges with smallest Q until either:
		// (a) the longest contiguous segment no longer satisfies the predicate
		// (b) the longest contiguous segment is <= 50% of the original track
		disconnects := make(map[int]bool)
		for {
			var bestQ float64 = 0.99
			var bestIdx int = -1
			for i, detection := range track {
				if disconnects[i] {
					continue
				}
				if detection.Score < bestQ {
					bestIdx = i
					bestQ = detection.Score
				}
			}
			if bestIdx == -1 {
				break
			}
			disconnects[bestIdx] = true
			segment := getLongestSegment(track, disconnects)
			if predFunc([][]miris.Detection{segment}) && len(segment) > len(track)/2 {
				continue
			}
			// it doesn't work, roll back this last disconnect and stop
			delete(disconnects, bestIdx)
			break
		}

		// largest Q threshold needed to correctly capture the track
		var trackMaxQ float64 = 1
		for i, detection := range track {
			if disconnects[i] {
				continue
			}
			if detection.Score < trackMaxQ {
				trackMaxQ = detection.Score
			}
		}
		samples = append(samples, trackMaxQ)
	}

	return samples
}

type qjob struct {
	freq    int
	segment miris.Segment
}

// Returns map from freq to list of MinQ samples.
func GetQSamples(maxFreq int, ppCfg miris.PreprocessConfig, modelCfg miris.ModelConfig) map[int][]float64 {
	predFunc := predicate.GetPredicate(ppCfg.Predicate)

	jobch := make(chan qjob)
	donech := make(chan map[int][]float64)

	for i := 0; i < QThreads; i++ {
		go func() {
			m := make(map[int][]float64)
			for job := range jobch {
				samples := GetQSamplesSegment(ppCfg, job.segment, job.freq, modelCfg.GetGNN(job.freq).ModelPath, predFunc)
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
		m := <-donech
		for freq, samples := range m {
			allSamples[freq] = append(allSamples[freq], samples...)
		}
	}
	return allSamples
}

func PlanQ(allSamples map[int][]float64, bound float64) map[int]float64 {
	freqToQ := map[int]float64{1: 1}
	for freq, samples := range allSamples {
		sort.Float64s(samples)
		q := samples[int((1-bound)*float64(len(samples)))]
		freqToQ[freq] = q
		log.Printf("[plan-q] compute q=%v at freq=%d", q, freq)
	}
	return freqToQ
}
