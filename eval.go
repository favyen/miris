package main

import (
	"./miris"
	"./predicate"

	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

// Returns frame idx corresponding to detection closest to track midpoint.
func midpointFrame(track []miris.Detection) int {
	midpoint := track[0].Bounds().Center().Add(track[len(track)-1].Bounds().Center()).Scale(0.5)
	var closestIdx int = -1
	var closestDistance float64
	for i, detection := range track {
		d := detection.Bounds().Center().Distance(midpoint)
		if closestIdx == -1 || d < closestDistance {
			closestIdx = i
			closestDistance = d
		}
	}
	return track[closestIdx].FrameIdx
}

func main() {
	gtFile := os.Args[1]
	trackFname := os.Args[2]

	gt := make(map[string]map[[2]int]int)
	counts := make(map[string]map[[2]int]int)

	bytes, err := ioutil.ReadFile(gtFile)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(bytes), "\n")
	var predNames []string
	var predFuncs []predicate.Predicate
	for _, predName := range strings.Split(strings.TrimSpace(lines[0]), ",")[1:] {
		predName = strings.TrimSpace(predName)
		predNames = append(predNames, predName)
		predFunc := predicate.GetPredicate(predName)
		if predFunc == nil {
			panic(fmt.Errorf("no predicate %s", predName))
		}
		predFuncs = append(predFuncs, predFunc)
		gt[predName] = make(map[[2]int]int)
		counts[predName] = make(map[[2]int]int)
	}
	for _, line := range lines[1:] {
		parts := strings.Split(strings.TrimSpace(line), ",")
		if parts[0] == "" {
			continue
		}
		rangeParts := strings.Split(parts[0], "-")
		lo, _ := strconv.Atoi(rangeParts[0])
		hi, _ := strconv.Atoi(rangeParts[1])
		r := [2]int{lo, hi}
		for i, s := range parts[1:] {
			val, _ := strconv.Atoi(s)
			gt[predNames[i]][r] = val
		}
	}

	var detections [][]miris.Detection
	miris.ReadJSON(trackFname, &detections)
	tracks := miris.GetTracks(detections)

	var fp int
	for _, track := range tracks {
		frameIdx := midpointFrame(track)

		for i, predFunc := range predFuncs {
			if !predFunc([][]miris.Detection{track}) {
				continue
			}
			predName := predNames[i]
			matched := false
			for r, gtval := range gt[predName] {
				if frameIdx < r[0] || frameIdx > r[1] {
					continue
				}
				if counts[predName][r] >= gtval {
					continue
				}
				counts[predName][r]++
				matched = true
			}
			if !matched {
				fp++
			}
			break
		}
	}

	var tp, fn int
	for predName := range gt {
		for r, gtval := range gt[predName] {
			tp += counts[predName][r]
			fn += gtval - counts[predName][r]
		}
	}

	var precision, recall, f1 float64
	if tp > 0 {
		precision = float64(tp)/float64(tp+fp)
		recall = float64(tp)/float64(tp+fn)
		f1 = 2/(1/precision+1/recall)
	}
	fmt.Printf("p=%v, r=%v, f=%v\n", precision, recall, f1)
}
