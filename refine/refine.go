package refine

import (
	"../miris"
	"../predicate"

	"log"
)

type RefinerFunc func(freq int, trainTracks [][]miris.Detection, predFunc predicate.Predicate, modelCfg map[string]string, cfg map[string]string) Refiner

type Refiner interface {
	Plan(valTracks [][]miris.Detection, bounds float64) map[string]string

	// returns list of frame indexes that need to be checked
	// and list of tracks that may need to be further refined
	Step(tracks [][]miris.Detection, seen []int) (needed []int, refined []int)
	Close()
}

var PSRefiners = make(map[string]RefinerFunc)
var InterpRefiners = make(map[string]RefinerFunc)

// Incorporate detections (that are labeled with track IDs) into tracks.
func incorporate(tracks map[int][]miris.Detection, detections []miris.Detection) {
	for _, detection := range detections {
		trackID := detection.TrackID
		track := tracks[trackID]
		insertIdx := 0
		for i, d := range track {
			if d.FrameIdx < detection.FrameIdx {
				insertIdx = i+1
			}
		}
		var newTrack []miris.Detection
		newTrack = append(newTrack, track[0:insertIdx]...)
		newTrack = append(newTrack, detection)
		newTrack = append(newTrack, track[insertIdx:]...)
		tracks[trackID] = newTrack
	}
}

// Runs refiners, given underlying detections that are labeled with track IDs.
// Returns list of frames examined, and the refined tracks.
func RunFake(refiners []Refiner, tracks [][]miris.Detection, detections [][]miris.Detection) ([]int, [][]miris.Detection) {
	seen := make(map[int]bool)
	for _, track := range tracks {
		for _, detection := range track {
			seen[detection.FrameIdx] = true
		}
	}

	getSeenList := func() []int {
		var seenList []int
		for frameIdx := range seen {
			seenList = append(seenList, frameIdx)
		}
		return seenList
	}

	trackByID := make(map[int][]miris.Detection)
	for _, track := range tracks {
		trackByID[track[0].TrackID] = track
	}

	for _, r := range refiners {
		var pending []int
		for trackID := range trackByID {
			pending = append(pending, trackID)
		}
		for len(pending) > 0 {
			inTracks := make([][]miris.Detection, len(pending))
			for i, trackID := range pending {
				inTracks[i] = trackByID[trackID]
			}
			needed, refined := r.Step(inTracks, getSeenList())
			log.Printf("[refine-runfake] ... need %d frames", len(needed))

			for _, frameIdx := range needed {
				if seen[frameIdx] {
					continue
				}
				if frameIdx < 0 || frameIdx >= len(detections) {
					seen[frameIdx] = true
					continue
				}
				incorporate(trackByID, detections[frameIdx])
				seen[frameIdx] = true
			}

			// map from refined to track IDs
			var nextPending []int
			for _, i := range refined {
				trackID := pending[i]
				nextPending = append(nextPending, trackID)
			}
			pending = nextPending
		}
	}

	var outTracks [][]miris.Detection
	for _, track := range trackByID {
		outTracks = append(outTracks, track)
	}

	return getSeenList(), outTracks
}
