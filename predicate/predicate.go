package predicate

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"../miris"
)

var predicates = make(map[string]Predicate)

type Predicate func(tracks [][]miris.Detection) bool

func GetPredicate(predicate string) Predicate {
	return predicates[predicate]
}

func StartEndPredicate(poly1 common.Polygon, poly2 common.Polygon) Predicate {
	return func(tracks [][]miris.Detection) bool {
		track := tracks[0]
		if len(track) == 0 {
			return false
		}
		return poly1.Contains(track[0].Bounds().Center()) && poly2.Contains(track[len(track)-1].Bounds().Center())
	}
}

// require track to pass through the polygons in any order
func PointSetPredicate(polygons []common.Polygon) Predicate {
	return func(tracks [][]miris.Detection) bool {
		track := tracks[0]
		for _, poly := range polygons {
			match := false
			for _, detection := range track {
				if !poly.Contains(detection.Bounds().Center()) {
					continue
				}
				match = true
				break
			}
			if !match {
				return false
			}
		}
		return true
	}
}

// track must pass through polygons in order
func WaypointPredicate(polygons []common.Polygon) Predicate {
	return func(tracks [][]miris.Detection) bool {
		track := tracks[0]
		polyIdx := 0
		for _, detection := range track {
			if !polygons[polyIdx].Contains(detection.Bounds().Center()) {
				continue
			}
			polyIdx++
			if polyIdx >= len(polygons) {
				break
			}
		}
		return polyIdx == len(polygons)
	}
}

func Or(predicates ...Predicate) Predicate {
	return func(tracks [][]miris.Detection) bool {
		for _, predicate := range predicates {
			if predicate(tracks) {
				return true
			}
		}
		return false
	}
}

// returns index of latest detection that precedes idx by at least (nframes)
func GetPredTime(track []miris.Detection, idx int, nframes int) int {
	for i := idx - 1; i >= 0; i-- {
		if track[i].FrameIdx < track[idx].FrameIdx - nframes {
			return i
		}
	}
	return -1
}

// returns index of closest predecessor to idx that is at least distance away
func GetPredDistance(track []miris.Detection, idx int, distance float64) int {
	for i := idx - 1 ; i >= 0; i-- {
		if track[i].Bounds().Center().Distance(track[idx].Bounds().Center()) >= distance {
			return i
		}
	}
	return -1
}
