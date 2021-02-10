package refine

import (
	"github.com/favyen/miris/miris"
	rnnlib "github.com/favyen/miris/models/rnn"
	"github.com/favyen/miris/predicate"

	"fmt"
	"sort"
	"strconv"
)

type RNNPSRefiner struct {
	freq      int
	predFunc  predicate.Predicate
	model     rnnlib.Model
	threshold float64
}

func MakeRNNPSRefiner(freq int, trainTracks [][]miris.Detection, predFunc predicate.Predicate, modelCfg map[string]string, cfg map[string]string) Refiner {
	model := rnnlib.MakeModel(2, modelCfg["model_path"])
	r := &RNNPSRefiner{
		freq:     freq,
		predFunc: predFunc,
		model:    model,
	}
	if cfg["threshold"] != "" {
		var err error
		r.threshold, err = strconv.ParseFloat(cfg["threshold"], 64)
		if err != nil {
			panic(err)
		}
	}
	return r
}

func init() {
	PSRefiners["rnn"] = MakeRNNPSRefiner
}

func (r *RNNPSRefiner) Plan(valTracks [][]miris.Detection, bound float64) map[string]string {
	type Example struct {
		Coarse   []miris.Detection
		Original map[int]miris.Detection
		Debug    []miris.Detection
		MaxT     float64
		PFreq    int
		SFreq    int
	}

	removeFakes := func(coarse []miris.Detection) []miris.Detection {
		var out []miris.Detection
		for _, detection := range coarse {
			if detection.Left == 0 && detection.Top == 0 && detection.Right == 0 && detection.Bottom == 0 {
				continue
			}
			out = append(out, detection)
		}
		return out
	}

	var samples []float64
	var remaining []Example
	for _, track := range valTracks {
		label := r.predFunc([][]miris.Detection{track})
		if !label {
			// negative->positive due to coarse prefix/suffix is unlikely
			continue
		}
		frameToDetection := make(map[int]miris.Detection)
		for _, detection := range track {
			frameToDetection[detection.FrameIdx] = detection
		}
		for k := 0; k < r.freq; k++ {
			coarse := GetCoarsePS(track, r.freq, k)
			if r.predFunc([][]miris.Detection{coarse}) == label {
				samples = append(samples, 1)
				continue
			} else if len(coarse) == 0 {
				samples = append(samples, 0)
				continue
			}
			remaining = append(remaining, Example{
				Coarse:   coarse,
				Original: frameToDetection,
				MaxT:     1,
				PFreq:    r.freq / 2,
				SFreq:    r.freq / 2,
				Debug:    track,
			})
		}
	}
	for len(remaining) > 0 {
		var tracks [][]miris.Detection
		for _, example := range remaining {
			tracks = append(tracks, example.Coarse)
		}
		outputs := r.model.Infer(tracks)
		var next []Example
		for i, example := range remaining {
			// add prefix/suffix depending on which is lower
			// unless one is freq=0, in which case we must add the other
			var doPrefix bool
			if example.SFreq == 0 {
				doPrefix = true
			} else if example.PFreq == 0 {
				doPrefix = false
			} else if outputs[i][0] > outputs[i][1] {
				doPrefix = true
			} else {
				doPrefix = false
			}
			withoutFakes := removeFakes(example.Coarse)
			if doPrefix {
				frameIdx := withoutFakes[0].FrameIdx - example.PFreq
				detection, ok := example.Original[frameIdx]
				if !ok {
					detection = miris.Detection{FrameIdx: frameIdx}
				}
				coarse := append([]miris.Detection{detection}, example.Coarse...)
				sort.Slice(coarse, func(i, j int) bool {
					return coarse[i].FrameIdx < coarse[j].FrameIdx
				})
				example.Coarse = coarse
				example.PFreq /= 2
				if outputs[i][0] < example.MaxT {
					example.MaxT = outputs[i][0]
				}
			} else {
				frameIdx := withoutFakes[len(withoutFakes)-1].FrameIdx + example.SFreq
				detection, ok := example.Original[frameIdx]
				if !ok {
					detection = miris.Detection{FrameIdx: frameIdx}
				}
				coarse := append([]miris.Detection{detection}, example.Coarse...)
				sort.Slice(coarse, func(i, j int) bool {
					return coarse[i].FrameIdx < coarse[j].FrameIdx
				})
				example.Coarse = coarse
				example.SFreq /= 2
				if outputs[i][1] < example.MaxT {
					example.MaxT = outputs[i][1]
				}
			}

			if r.predFunc([][]miris.Detection{removeFakes(example.Coarse)}) {
				samples = append(samples, example.MaxT)
				continue
			} else if example.SFreq == 0 && example.PFreq == 0 {
				samples = append(samples, 0)
				continue
			}

			next = append(next, example)
		}
		remaining = next
	}

	sort.Float64s(samples)
	r.threshold = samples[int((1-bound)*float64(len(samples)))]
	return map[string]string{
		"threshold": fmt.Sprintf("%v", r.threshold),
	}
}

func (r *RNNPSRefiner) Step(tracks [][]miris.Detection, seen []int) ([]int, []int) {
	seenSet := make(map[int]bool)
	for _, frameIdx := range seen {
		seenSet[frameIdx] = true
	}

	getFreq := func(frameIdx int) int {
		for freq := r.freq; freq >= 2; freq /= 2 {
			if frameIdx%freq == 0 {
				return freq
			}
		}
		return 1
	}

	var checkTracks [][]miris.Detection
	var checkFrames [][2]int
	for _, track := range tracks {
		if r.predFunc([][]miris.Detection{track}) || len(track) == 0 {
			continue
		}

		// add fake detections to prefix/suffix if the track is missing in those frames
		curFrames := [2]int{-1, -1}
		pFreq := getFreq(track[0].FrameIdx)
		for freq := pFreq / 2; freq >= 1; freq /= 2 {
			frameIdx := track[0].FrameIdx - freq
			if !seenSet[frameIdx] {
				curFrames[0] = frameIdx
				break
			}
			fake := miris.Detection{FrameIdx: frameIdx}
			track = append([]miris.Detection{fake}, track...)
		}
		sFreq := getFreq(track[len(track)-1].FrameIdx)
		for freq := sFreq / 2; freq >= 1; freq /= 2 {
			frameIdx := track[len(track)-1].FrameIdx + freq
			if !seenSet[frameIdx] {
				curFrames[1] = frameIdx
				break
			}
			fake := miris.Detection{FrameIdx: frameIdx}
			track = append(track, fake)
		}
		if curFrames[0] == -1 && curFrames[1] == -1 {
			continue
		}
		checkTracks = append(checkTracks, track)
		checkFrames = append(checkFrames, curFrames)
	}

	if len(checkTracks) == 0 {
		return nil, nil
	}

	outputs := r.model.Infer(checkTracks)
	needed := make(map[int]bool)
	var refined []int
	for i, output := range outputs {
		if (output[0] < r.threshold || checkFrames[i][0] == -1) && (output[1] < r.threshold || checkFrames[i][1] == -1) {
			continue
		}
		refined = append(refined, i)
		if output[0] >= r.threshold && checkFrames[i][0] != -1 {
			needed[checkFrames[i][0]] = true
		}
		if output[1] >= r.threshold && checkFrames[i][1] != -1 {
			needed[checkFrames[i][1]] = true
		}
	}

	var neededList []int
	for frameIdx := range needed {
		neededList = append(neededList, frameIdx)
	}
	return neededList, refined
}

func (r *RNNPSRefiner) Close() {
	r.model.Close()
}
