package main

import (
	"./data"
	"./miris"
	"./planner"

	"fmt"
	"log"
	"os"
	"strconv"
)

func main() {
	predName := os.Args[1]
	freq, _ := strconv.Atoi(os.Args[2])
	bound, _ := strconv.ParseFloat(os.Args[3], 64)

	var existingPlan miris.PlannerConfig
	var qSamples map[int][]float64
	if len(os.Args) >= 5 {
		miris.ReadJSON(os.Args[4], &existingPlan)
		qSamples = existingPlan.QSamples
	}

	ppCfg, modelCfg := data.Get(predName)

	if qSamples == nil {
		qSamples = planner.GetQSamples(2*freq, ppCfg, modelCfg)
	}
	q := planner.PlanQ(qSamples, bound)
	log.Println("finished planning q", q)
	plan := miris.PlannerConfig{
		Freq: freq,
		Bound: bound,
		QSamples: qSamples,
		Q: q,
	}
	miris.WriteJSON(fmt.Sprintf("logs/%s/%d/%v/plan.json", predName, freq, bound), plan)
	filterPlan, refinePlan := planner.PlanFilterRefine(ppCfg, modelCfg, freq, bound, nil)
	plan.Filter = filterPlan
	plan.Refine = refinePlan
	log.Println(plan)
	miris.WriteJSON(fmt.Sprintf("logs/%s/%d/%v/plan.json", predName, freq, bound), plan)
}

