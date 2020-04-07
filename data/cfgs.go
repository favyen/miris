package data

import (
	"../miris"

	"fmt"
)

func Get(predName string) (miris.PreprocessConfig, miris.ModelConfig) {
	if predName == "shibuya" || predName == "shibuya-cx" {
		return Shibuya(predName)
	} else if predName == "warsaw" || predName == "warsaw-brake" {
		return Warsaw(predName)
	} else if predName == "beach-runner" {
		return Beach(predName)
	}
	panic(fmt.Errorf("unknown predicate %s", predName))
}

func GetExec(predName string) (detectionPath string, framePath string) {
	if predName == "shibuya" || predName == "shibuya-cx" {
		detectionPath = "data/shibuya/json/2-detections.json"
		framePath = "data/shibuya/frames/2/"
	} else if predName == "warsaw" || predName == "warsaw-brake" {
		detectionPath = "data/warsaw/json/2-detections.json"
		framePath = "data/warsaw/frames/2/"
	} else if predName == "beach-runner" {
		detectionPath = "data/beach/json/2-detections.json"
		framePath = "data/beach/frames/2/"
	} else {
		panic(fmt.Errorf("unknown predicate %s", predName))
	}
	return
}

func Shibuya(predName string) (miris.PreprocessConfig, miris.ModelConfig) {
	var segments []miris.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, miris.Segment{
			FramePath: fmt.Sprintf("data/shibuya/frames/%d/", i),
			TrackPath: fmt.Sprintf("data/shibuya/json/%d-baseline.json", i),
		})
	}

	ppCfg := miris.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments: segments[2:],
		Predicate: predName,
		FrameScale: 2,
	}
	var modelCfg miris.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, miris.GNNModel{
			Freq: freq,
			ModelPath: "logs/shibuya/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, miris.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, miris.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	return ppCfg, modelCfg
}

func Warsaw(predName string) (miris.PreprocessConfig, miris.ModelConfig) {
	var segments []miris.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, miris.Segment{
			FramePath: fmt.Sprintf("data/warsaw/frames/%d/", i),
			TrackPath: fmt.Sprintf("data/warsaw/json/%d-baseline.json", i),
		})
	}

	ppCfg := miris.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments: segments[2:],
		Predicate: predName,
		FrameScale: 2,
	}
	var modelCfg miris.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, miris.GNNModel{
			Freq: freq,
			ModelPath: "logs/warsaw/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, miris.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, miris.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	return ppCfg, modelCfg
}

func Beach(predName string) (miris.PreprocessConfig, miris.ModelConfig) {
	var segments []miris.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, miris.Segment{
			FramePath: fmt.Sprintf("data/beach/frames/%d/", i),
			TrackPath: fmt.Sprintf("data/beach/json/%d-track-res960-freq1.json", i),
		})
	}

	ppCfg := miris.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments: segments[2:],
		Predicate: predName,
		FrameScale: 2,
	}
	var modelCfg miris.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, miris.GNNModel{
			Freq: freq,
			ModelPath: "logs/beach/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, miris.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, miris.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	return ppCfg, modelCfg
}
