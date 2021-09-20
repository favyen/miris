package planner

import (
	"github.com/favyen/miris/filter"
	"github.com/favyen/miris/miris"
	"github.com/favyen/miris/refine"

	"log"
)

func PlanRefine(context plannerContext, filterPlan miris.FilterPlan) miris.RefinePlan {
	psRefiners := make(map[string]refine.Refiner)
	interpRefiners := make(map[string]refine.Refiner)
	refinerConfigs := make(map[string]map[string]string)
	for name, refinerFunc := range refine.PSRefiners {
		log.Printf("[plan-refine] trying ps refiner %s", name)
		r := refinerFunc(context.freq, context.trainTracks, context.predFunc, context.modelCfg.GetRefineCfg(name, context.freq), nil)
		defer r.Close()
		cfg := r.Plan(context.valTracks, context.bound)
		log.Printf("[plan-refine] ... got cfg=%v", cfg)
		psRefiners[name] = r
		refinerConfigs[name] = cfg
	}
	for name, refinerFunc := range refine.InterpRefiners {
		log.Printf("[plan-refine] trying interp refiner %s", name)
		r := refinerFunc(context.freq, context.trainTracks, context.predFunc, context.modelCfg.GetRefineCfg(name, context.freq), nil)
		defer r.Close()
		cfg := r.Plan(context.valTracks, context.bound)
		log.Printf("[plan-refine] ... got cfg=%v", cfg)
		interpRefiners[name] = r
		refinerConfigs[name] = cfg
	}

	// determine how many frames each refiner wants to look at when
	// refining the val tracks after filtering
	trainLabels := make([]bool, len(context.trainTracks))
	for i, track := range context.trainTracks {
		trainLabels[i] = context.predFunc([][]miris.Detection{track})
	}
	selFilter := filter.FilterMap[filterPlan.Name](context.freq, context.trainTracks, trainLabels, context.modelCfg.GetFilterCfg(filterPlan.Name, context.freq))
	coarseTracks := make([][]miris.Detection, len(context.valTracks))
	for i, track := range context.valTracks {
		coarseTracks[i] = miris.GetCoarse(track, context.freq, 0)
	}
	scores := selFilter.Predict(coarseTracks)
	selFilter.Close()
	var filteredTracks [][]miris.Detection
	var origSatisfy, filterSatisfy int
	for i, track := range context.valTracks {
		satisfy := context.predFunc([][]miris.Detection{track})
		if satisfy {
			origSatisfy++
		}
		if scores[i] < filterPlan.Threshold || len(coarseTracks[i]) == 0 {
			continue
		}
		if satisfy {
			filterSatisfy++
		}
		filteredTracks = append(filteredTracks, coarseTracks[i])
	}
	log.Printf("[plan-refine] %d/%d tracks passed filtering (%d satisfy before filter, %d after)", len(filteredTracks), len(context.valTracks), origSatisfy, filterSatisfy)

	var bestNames [2]string
	var bestFrames int = -1
	for name1, r1 := range psRefiners {
		for name2, r2 := range interpRefiners {
			log.Printf("[plan-refine] measuring # frames for (%s, %s)", name1, name2)
			refiners := []refine.Refiner{r1, r2}
			seen, _ := refine.RunFake(refiners, filteredTracks, context.valDetections)
			log.Printf("[plan-refine] ... (%s, %s) used %d frames", name1, name2, len(seen))
			if bestFrames == -1 || len(seen) < bestFrames {
				bestFrames = len(seen)
				bestNames = [2]string{name1, name2}
			}
		}
	}
	log.Printf("[plan-refine] decided to use (%s, %s)", bestNames[0], bestNames[1])
	return miris.RefinePlan{
		PSMethod:     bestNames[0],
		PSCfg:        refinerConfigs[bestNames[0]],
		InterpMethod: bestNames[1],
		InterpCfg:    refinerConfigs[bestNames[1]],
	}
}
