package refine

import (
	"../miris"
	"../predicate"

	"fmt"
	"strconv"
	"sort"
)

func GetCoarsePS(freq int, k int, track []miris.Detection) []miris.Detection {
	start := -1
	end := -1
	for i, detection := range track {
		if detection.FrameIdx % freq != k {
			continue
		}
		if start == -1 {
			start = i
		}
		end = i
	}
	if start == -1 || end == -1 {
		return nil
	}
	return track[start:end+1]
}

type SimplePSRefiner struct {
	freq int
	predFunc predicate.Predicate
	freqThreshold int
}

func MakeSimplePSRefiner(freq int, trainTracks [][]miris.Detection, predFunc predicate.Predicate, cfg map[string]string) Refiner {
	r := &SimplePSRefiner{
		freq: freq,
		predFunc: predFunc,
	}
	if cfg["threshold"] != "" {
		var err error
		r.freqThreshold, err = strconv.Atoi(cfg["threshold"])
		if err != nil {
			panic(err)
		}
	}
	return r
}

func init() {
	PSRefiners["simple"] = MakeSimplePSRefiner
}

func (r *SimplePSRefiner) Plan(valTracks [][]miris.Detection, bound float64) map[string]string {
	// for each coarse track, find the freqThreshold needed to get the predicate correct
	// (but retain all intermediate detections in the coarse tracks)
	// then choose a threshold based on bounds
	var samples []int
	for _, track := range valTracks {
		label := r.predFunc([][]miris.Detection{track})
		if !label {
			// negative->positive due to coarse is unlikely
			continue
		}
		for k := 0; k < r.freq; k++ {
			freqThreshold := r.freq
			for {
				coarse := GetCoarsePS(freqThreshold, k%freqThreshold, track)
				if r.predFunc([][]miris.Detection{coarse}) == label {
					break
				}
				freqThreshold /= 2
				if freqThreshold < 1 {
					panic(fmt.Errorf("simple ps planner: freqThreshold==1 should always succeed"))
				}
			}
			samples = append(samples, freqThreshold)
		}
	}
	sort.Ints(samples)
	r.freqThreshold = samples[int((1-bound)*float64(len(samples)))]
	return map[string]string{
		"threshold": fmt.Sprintf("%d", r.freqThreshold),
	}
}

func (r *SimplePSRefiner) Step(tracks [][]miris.Detection, seen []int) ([]int, []int) {
	seenSet := make(map[int]bool)
	for _, frameIdx := range seen {
		seenSet[frameIdx] = true
	}

	getFreq := func(frameIdx int) int {
		for _, freq := range []int{16, 8, 4, 2} {
			if frameIdx % freq == 0 {
				return freq
			}
		}
		return 1
	}

	// Get the next frame idx that we need to look at.
	find := func(frameIdx int, direction int) int {
		freq := getFreq(frameIdx)
		for seenSet[frameIdx] {
			freq = freq / 2
			if freq < r.freqThreshold {
				return -1
			}
			frameIdx = frameIdx + direction*freq
		}
		return frameIdx
	}

	needed := make(map[int]bool)
	var refined []int
	for i, track := range tracks {
		prefixIdx := find(track[0].FrameIdx, -1)
		suffixIdx := find(track[len(track)-1].FrameIdx, 1)
		if prefixIdx != -1 {
			needed[prefixIdx] = true
		}
		if suffixIdx != -1 {
			needed[suffixIdx] = true
		}
		if prefixIdx != -1 || suffixIdx != -1 {
			refined = append(refined, i)
		}
	}
	var neededList []int
	for frameIdx := range needed {
		neededList = append(neededList, frameIdx)
	}
	return neededList, refined
}

func (r *SimplePSRefiner) Close() {}
