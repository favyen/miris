package predicate

// miscellaneous queries

import (
	"../miris"
	"github.com/mitroadmaps/gomapinfer/common"
)

func init() {
	predicates["warsaw-brake"] = WarsawBrake
}

func WarsawBrake(tracks [][]miris.Detection) bool {
	track := tracks[0]
	if len(track) == 0 {
		return false
	}

	rect1 := common.Rectangle{
		common.Point{0, 610},
		common.Point{930, 1080},
	}
	rect2 := common.Rectangle{
		common.Point{1190, 700},
		common.Point{1920, 1080},
	}
	if !rect1.Contains(track[0].Bounds().Center()) || !rect2.Contains(track[len(track)-1].Bounds().Center()) {
		return false
	}


	// need to have hard braking: high speed in 5 frames, then 5 frames, then zero in 5 frames
	// high speed = 300 px / 5 frames, zero = 30 px / 5 frames
	for i := range track {
		zeroIdx := GetPredTime(track, i, 5)
		brakeIdx := GetPredTime(track, zeroIdx, 30)
		highIdx := GetPredTime(track, brakeIdx, 5)
		if highIdx == -1 {
			continue
		}
		zeroSpeed := track[zeroIdx].Bounds().Center().Distance(track[i].Bounds().Center())
		highSpeed := track[highIdx].Bounds().Center().Distance(track[brakeIdx].Bounds().Center())
		if zeroSpeed < 30 && highSpeed > 200 {
			return true
		}
	}
	return false
}
