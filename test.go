package main

import (
	"./exec"
	"./miris"
	"./planner"

	"fmt"
	"log"
)

func main() {
	var segments []miris.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, miris.Segment{
			FramePath: fmt.Sprintf("/data2/youtube/shibuya/frames-half/%d/", i),
			TrackPath: fmt.Sprintf("/data2/youtube/shibuya/json/%d-track-res960-freq1.json", i),
		})
	}

	ppCfg := miris.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments: segments[2:],
		Predicate: "shibuya",
		FrameScale: 2,
	}
	modelCfg := miris.ModelConfig{
		GNN: []miris.GNNModel{
			{Freq: 32, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
			{Freq: 16, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
			{Freq: 8, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
			{Freq: 4, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
			{Freq: 2, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
			{Freq: 1, ModelPath: "/mnt/dji/miris-models/shibuya/m19-skipvar/model_best/model"},
		},
		Filters: []miris.FilterModel{{
			Name: "rnn",
			Freq: 16,
			Cfg: map[string]string{
				//"model_path": "/mnt/dji/miris-models/shibuya/filter-rnn16/model_best/model",
				"model_path": "/home/ubuntu/miris2/logs/16/filter-rnn/model",
			},
		}},
		Refiners: []miris.RefineModel{{
			Name: "rnn",
			Freq: 16,
			Cfg: map[string]string{
				"model_path": "/home/ubuntu/miris2/logs/16/refine-rnn/model",
			},
		}},
	}
	if false {
		var origPlan miris.PlannerConfig
		miris.ReadJSON("plan.json", &origPlan)

		freq := 16
		bound := 0.95
		//q := planner.PlanQ(2*freq, bound, ppCfg, modelCfg)
		q := map[int]float64{
			1: 1, 2: 1, 4: 1, 8: 1, 16: 1, 32: 0.69,
		}
		filterPlan, refinePlan := planner.PlanFilterRefine(ppCfg, modelCfg, freq, bound, &origPlan.Filter)
		plan := miris.PlannerConfig{
			Freq: freq,
			Filter: filterPlan,
			Q: q,
			Refine: refinePlan,
		}
		log.Println(plan)
		miris.WriteJSON("plan.json", plan)
		return
	}

	var plan miris.PlannerConfig
	miris.ReadJSON("plan2.json", &plan)
	execCfg := miris.ExecConfig{
		DetectionPath: "/data2/youtube/shibuya/json/2-detect-res960.json",
		FramePath: "/data2/youtube/shibuya/frames-half/2/",
		TrackOutput: "/tmp/track.json",
		FilterOutput: "/tmp/filter.json",
		UncertaintyOutput: "/tmp/uncertainty.json",
		RefineOutput: "/tmp/refine.json",
		OutPath: "/tmp/out.json",
	}
	exec.Exec(ppCfg, modelCfg, plan, execCfg)
}

