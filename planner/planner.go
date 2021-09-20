package planner

import (
	"github.com/favyen/miris/miris"
	"github.com/favyen/miris/predicate"

	"log"
)

type plannerContext struct {
	ppCfg           miris.PreprocessConfig
	modelCfg        miris.ModelConfig
	freq            int
	bound           float64
	trainDetections [][]miris.Detection
	trainTracks     [][]miris.Detection
	valDetections   [][]miris.Detection
	valTracks       [][]miris.Detection
	predFunc        predicate.Predicate
}

func PlanFilterRefine(ppCfg miris.PreprocessConfig, modelCfg miris.ModelConfig, freq int, bound float64, existingFilterPlan *miris.FilterPlan) (miris.FilterPlan, miris.RefinePlan) {
	context := plannerContext{
		ppCfg:    ppCfg,
		modelCfg: modelCfg,
		freq:     freq,
		bound:    bound,
	}

	increment := func(detections [][]miris.Detection, frames int, trackID int) {
		for _, dlist := range detections {
			for i := range dlist {
				dlist[i].FrameIdx += frames
				dlist[i].TrackID += trackID
			}
		}
	}
	getMaxTrackID := func(tracks [][]miris.Detection) int {
		max := 0
		for _, track := range tracks {
			if track[0].TrackID > max {
				max = track[0].TrackID
			}
		}
		return max
	}

	context.predFunc = predicate.GetPredicate(ppCfg.Predicate)
	log.Printf("[planner] loading train tracks")
	for _, segment := range ppCfg.TrainSegments {
		segDetections := miris.ReadDetections(segment.TrackPath)
		segTracks := miris.GetTracks(segDetections)
		increment(segDetections, len(context.trainDetections), 0)
		increment(segTracks, len(context.trainDetections), getMaxTrackID(context.trainTracks)+1)
		context.trainDetections = append(context.trainDetections, segDetections...)
		context.trainTracks = append(context.trainTracks, segTracks...)
	}
	log.Printf("[planner] loading val tracks")
	for _, segment := range ppCfg.ValSegments {
		segDetections := miris.ReadDetections(segment.TrackPath)
		segTracks := miris.GetTracks(segDetections)
		increment(segDetections, len(context.valDetections), 0)
		increment(segTracks, len(context.valDetections), getMaxTrackID(context.valTracks)+1)
		context.valDetections = append(context.valDetections, segDetections...)
		context.valTracks = append(context.valTracks, segTracks...)
	}

	var filterPlan miris.FilterPlan
	if existingFilterPlan != nil {
		filterPlan = *existingFilterPlan
	} else {
		filterPlan = PlanFilter(context)
	}
	//filterPlan := miris.FilterPlan{"nnd", -18.007}
	refinePlan := PlanRefine(context, filterPlan)

	return filterPlan, refinePlan
}
