package planner

import (
	"../miris"
	"../predicate"

	"log"
)

type plannerContext struct {
	ppCfg miris.PreprocessConfig
	modelCfg miris.ModelConfig
	freq int
	bound float64
	trainDetections [][]miris.Detection
	trainTracks [][]miris.Detection
	valDetections [][]miris.Detection
	valTracks [][]miris.Detection
	predFunc predicate.Predicate
}

func PlanFilterRefine(ppCfg miris.PreprocessConfig, modelCfg miris.ModelConfig, freq int, bound float64) (miris.FilterPlan, miris.RefinePlan) {
	context := plannerContext{
		ppCfg: ppCfg,
		modelCfg: modelCfg,
		freq: freq,
		bound: bound,
	}

	context.predFunc = predicate.GetPredicate(ppCfg.Predicate)
	log.Printf("[planner] loading train tracks")
	for _, segment := range ppCfg.TrainSegments {
		segDetections := miris.ReadDetections(segment.TrackPath)
		segTracks := miris.GetTracks(segDetections)
		context.trainDetections = append(context.trainDetections, segDetections...)
		context.trainTracks = append(context.trainTracks, segTracks...)
	}
	log.Printf("[planner] loading val tracks")
	for _, segment := range ppCfg.ValSegments {
		segDetections := miris.ReadDetections(segment.TrackPath)
		segTracks := miris.GetTracks(segDetections)
		context.valDetections = append(context.valDetections, segDetections...)
		context.valTracks = append(context.valTracks, segTracks...)
	}

	filterPlan := PlanFilter(context)
	//filterPlan := miris.FilterPlan{"nnd", -18.007}
	refinePlan := PlanRefine(context, filterPlan)

	return filterPlan, refinePlan
}
