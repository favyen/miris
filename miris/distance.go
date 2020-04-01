package miris

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/spmetric"
)

func SamplePoints(track []Detection) []common.Point {
	var points []common.Point
	for i := 0; i < len(track) - 1; i++ {
		segment := common.Segment{track[i].Bounds().Center(), track[i+1].Bounds().Center()}
		points = append(points, segment.Sample(10)...)
	}
	return points
}

func SampleNormalizedPoints(track []Detection) []common.Point {
	// sample twenty points along the track
	var trackLength float64 = 0
	for i := 0; i < len(track) - 1; i++ {
		trackLength += track[i].Bounds().Center().Distance(track[i+1].Bounds().Center())
	}
	pointFreq := trackLength / 20

	points := []common.Point{track[0].Bounds().Center()}
	remaining := pointFreq
	for i := 0; i < len(track) - 1; i++ {
		segment := common.Segment{track[i].Bounds().Center(), track[i+1].Bounds().Center()}
		for segment.Length() > remaining {
			vector := segment.Vector()
			p := segment.Start.Add(vector.Scale(remaining / segment.Length()))
			points = append(points, p)
			segment = common.Segment{p, segment.End}
			remaining = pointFreq
		}
		remaining -= segment.Length()
	}
	for len(points) < 20 {
		points = append(points, track[len(track)-1].Bounds().Center())
	}
	return points[0:20]
}

func TrackDistanceLowerBound(track1 []Detection, track2 []Detection, threshold float64) bool {
	return false
	for _, d1 := range track1 {
		p1 := d1.Bounds().Center()
		var bestDistance float64 = -1
		for _, d2 := range track2 {
			p2 := d2.Bounds().Center()
			d := p1.Distance(p2)
			if bestDistance == -1 || d < bestDistance {
				bestDistance = d
			}
		}
		if bestDistance > threshold {
			return true
		}
	}
	return false
}

/*func TrackDistance(track1 []Detection, track2 []Detection) float64 {
	points1 := SampleNormalizedPoints(track1)
	points2 := SampleNormalizedPoints(track2)
	var maxDistance float64 = 0
	for i := range points1 {
		d := points1[i].Distance(points2[i])
		if d > maxDistance {
			maxDistance = d
		}
	}
	return maxDistance
}*/

func TrackDistance(track1 []Detection, track2 []Detection) float64 {
	// track1 is coarse, track2 is dense
	// it's like frechet distance but we only look at distance from points in track1
	// and do it greedily

	// get closest detection idx in track2
	getClosestIdxAfter := func(p common.Point, j int) int {
		bestIdx := j
		var bestDistance float64 = -1
		for i := j; i < len(track2); i++ {
			d := p.Distance(track2[i].Bounds().Center())
			if bestDistance == -1 || d < bestDistance {
				bestIdx = i
				bestDistance = d
			}
		}
		return bestIdx
	}

	var maxDistance float64
	idx2 := 0
	for _, detection := range track1 {
		p1 := detection.Bounds().Center()
		idx2 = getClosestIdxAfter(p1, idx2)
		p2 := track2[idx2].Bounds().Center()
		d := p1.Distance(p2)
		if d > maxDistance {
			maxDistance = d
		}
	}

	return maxDistance
}

func FrechetDistance(track1 []Detection, track2 []Detection) float64 {
	points1 := SamplePoints(track1)
	points2 := SamplePoints(track2)
	return spmetric.ComputeFrechetDistance(points1, points2)
}
