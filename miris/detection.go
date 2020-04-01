package miris

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"io/ioutil"
	"sort"
)

type Detection struct {
	FrameIdx int `json:"frame_idx"`
	TrackID int `json:"track_id"`
	Left int `json:"left"`
	Top int `json:"top"`
	Right int `json:"right"`
	Bottom int `json:"bottom"`
	Score float64 `json:"score,omitempty"`
}

func (d Detection) Bounds() common.Rectangle {
	return common.Rectangle{
		Min: common.Point{float64(d.Left), float64(d.Top)},
		Max: common.Point{float64(d.Right), float64(d.Bottom)},
	}
}

func (d Detection) Equals(other Detection) bool {
	return d.FrameIdx == other.FrameIdx && d.Left == other.Left && d.Top == other.Top && d.Right == other.Right && d.Bottom == other.Bottom
}

func Interpolate(a Detection, b Detection, frameIdx int) Detection {
	factor := float64(frameIdx - a.FrameIdx) / float64(b.FrameIdx - a.FrameIdx)
	d := Detection{
		FrameIdx: frameIdx,
		TrackID: a.TrackID,
	}
	d.Left = int(factor * float64(b.Left - a.Left)) + a.Left
	d.Top = int(factor * float64(b.Top - a.Top)) + a.Top
	d.Right = int(factor * float64(b.Right - a.Right)) + a.Right
	d.Bottom = int(factor * float64(b.Bottom - a.Bottom)) + a.Bottom
	return d
}

func Densify(track []Detection) []Detection {
	var denseTrack []Detection
	for _, detection := range track {
		if len(denseTrack) > 0 {
			prev := denseTrack[len(denseTrack) - 1]
			for frameIdx := prev.FrameIdx + 1; frameIdx < detection.FrameIdx; frameIdx++ {
				denseTrack = append(denseTrack, Interpolate(prev, detection, frameIdx))
			}
		}
		denseTrack = append(denseTrack, detection)
	}
	return denseTrack
}

func DensifyAt(track []Detection, indexes []int) []Detection {
	relevant := make(map[int]bool)
	for _, frameIdx := range indexes {
		if frameIdx < track[0].FrameIdx || frameIdx > track[len(track)-1].FrameIdx {
			continue
		}
		relevant[frameIdx] = true
	}
	for _, detection := range track {
		delete(relevant, detection.FrameIdx)
	}
	var denseTrack []Detection
	for _, detection := range track {
		if len(denseTrack) > 0 {
			prev := denseTrack[len(denseTrack) - 1]
			for frameIdx := prev.FrameIdx + 1; frameIdx < detection.FrameIdx; frameIdx++ {
				if !relevant[frameIdx] {
					continue
				}
				denseTrack = append(denseTrack, Interpolate(prev, detection, frameIdx))
			}
		}
		denseTrack = append(denseTrack, detection)
	}
	return denseTrack
}

type FeatureVector [64]float64
func (v1 FeatureVector) Distance(v2 FeatureVector) float64 {
	var d float64 = 0
	for i := 0; i < len(v1); i++ {
		d += (v1[i] - v2[i]) * (v1[i] - v2[i])
	}
	return d / float64(len(v1))
}

type ActionVectorJSON struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	P float64 `json:"p"`
}
func (v ActionVectorJSON) ActionVector() ActionVector {
	return ActionVector{
		Displacement: common.Point{v.X, v.Y},
		Probability: v.P,
	}
}

type ActionVector struct {
	Displacement common.Point

	// probability that the track remains in the frame
	Probability float64
}

func ReadDetections(fname string) [][]Detection {
	bytes, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	var detections [][]Detection
	if err := json.Unmarshal(bytes, &detections); err != nil {
		panic(err)
	}
	return detections
}

func GetTracks(detections [][]Detection) [][]Detection {
	tracks := make(map[int][]Detection)
	for frameIdx := range detections {
		for _, detection := range detections[frameIdx] {
			if detection.TrackID < 0 {
				continue
			}
			tracks[detection.TrackID] = append(tracks[detection.TrackID], detection)
		}
	}
	var trackList [][]Detection
	for _, track := range tracks {
		trackList = append(trackList, track)
	}
	sort.Slice(trackList, func(i, j int) bool {
		return trackList[i][0].FrameIdx < trackList[j][0].FrameIdx
	})
	return trackList
}

func TracksToDetections(tracks [][]Detection) [][]Detection {
	var detections [][]Detection
	for _, track := range tracks {
		for _, detection := range track {
			for len(detections) <= detection.FrameIdx {
				detections = append(detections, []Detection{})
			}
			detections[detection.FrameIdx] = append(detections[detection.FrameIdx], detection)
		}
	}
	return detections
}

func FilterByScore(detections [][]Detection, threshold float64) [][]Detection {
	ndetections := make([][]Detection, len(detections))
	for frameIdx, dlist := range detections {
		for _, detection := range dlist {
			if detection.Score < threshold {
				continue
			}
			ndetections[frameIdx] = append(ndetections[frameIdx], detection)
		}
	}
	return ndetections
}

func CountDetections(detections [][]Detection) int {
	var n int = 0
	for _, dlist := range detections {
		n += len(dlist)
	}
	return n
}

func GetCoarse(track []Detection, freq int, k int) []Detection {
	var coarse []Detection
	for _, detection := range track {
		if detection.FrameIdx % freq != k {
			continue
		}
		coarse = append(coarse, detection)
	}
	return coarse
}

func GetAllCoarse(track []Detection, freq int) [][]Detection {
	var l [][]Detection
	for k := 0; k < freq; k++ {
		l = append(l, GetCoarse(track, freq, k))
	}
	return l
}
