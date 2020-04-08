package predicate

// miscellaneous queries

import (
	"../miris"
	"github.com/mitroadmaps/gomapinfer/common"
)

func init() {
	predicates["warsaw-brake"] = WarsawBrake
	predicates["beach-runner"] = BeachRunner
	predicates["shibuya-crosswalk"] = ShibuyaCrosswalk
}

func WarsawBrake(tracks [][]miris.Detection) bool {
	track := miris.Densify(tracks[0])
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

func BeachRunner(tracks [][]miris.Detection) bool {
	track := miris.Densify(tracks[0])
	poly := common.Polygon{
		common.Point{153, 676},
		common.Point{520, 990},
		common.Point{1083, 990},
		common.Point{1369, 640},
		common.Point{760, 535},
	}
	for i, detection := range track {
		p1 := detection.Bounds().Center()
		if !poly.Contains(p1) {
			continue
		}
		j := GetPredDistance(track, i, 350)
		if j == -1 {
			continue
		}
		p2 := track[j].Bounds().Center()
		if !poly.Contains(p2) {
			continue
		}
		if track[i].FrameIdx - track[j].FrameIdx <= 20 {
			return true
		}
	}
	return false
}

func ShibuyaCrosswalk(tracks [][]miris.Detection) bool {
	track := tracks[0]
	// stopped for > 15 sec (150 frames) in crosswalk
	poly := common.Polygon{
		common.Point{519, 463},
		common.Point{115, 660},
		common.Point{129, 987},
		common.Point{1210, 1071},
		common.Point{1911, 1072},
		common.Point{1909, 826},
		common.Point{1656, 670},
		common.Point{1642, 558},
		common.Point{1696, 471},
		common.Point{1438, 406},
		common.Point{1338, 420},
		common.Point{1162, 408},
		common.Point{1027, 393},
	}
	for i, detection := range track {
		p1 := detection.Bounds().Center()
		if !poly.Contains(p1) {
			continue
		}
		j := GetPredTime(track, i, 150)
		if j == -1 {
			continue
		}
		p2 := track[j].Bounds().Center()
		if !poly.Contains(p2) {
			continue
		}
		if p1.Distance(p2) < 30 {
			return true
		}
	}
	return false
}
