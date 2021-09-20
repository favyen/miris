package miris

import (
	"fmt"
)

type Segment struct {
	FramePath string
	TrackPath string
}

type PreprocessConfig struct {
	TrainSegments []Segment
	ValSegments   []Segment
	Predicate     string
	FrameScale    int
}

type FilterModel struct {
	Name string
	Freq int
	Cfg  map[string]string
}

type RefineModel struct {
	Name string
	Freq int
	Cfg  map[string]string
}

type GNNModel struct {
	Freq      int
	ModelPath string
}

type ModelConfig struct {
	Filters  []FilterModel
	Refiners []RefineModel
	GNN      []GNNModel
}

func (cfg ModelConfig) GetFilterCfg(name string, freq int) map[string]string {
	for _, filter := range cfg.Filters {
		if filter.Name != name || filter.Freq != freq {
			continue
		}
		return filter.Cfg
	}
	return nil
}

func (cfg ModelConfig) GetRefineCfg(name string, freq int) map[string]string {
	for _, refiner := range cfg.Refiners {
		if refiner.Name != name || refiner.Freq != freq {
			continue
		}
		return refiner.Cfg
	}
	return nil
}

func (cfg ModelConfig) GetGNN(freq int) GNNModel {
	for _, gcfg := range cfg.GNN {
		if gcfg.Freq == freq {
			return gcfg
		}
	}
	panic(fmt.Errorf("no gnn model cfg at freq %d", freq))
}

type FilterPlan struct {
	Name      string
	Threshold float64
}

type RefinePlan struct {
	PSMethod     string
	PSCfg        map[string]string
	InterpMethod string
	InterpCfg    map[string]string
}

type PlannerConfig struct {
	Freq     int
	Bound    float64
	Filter   FilterPlan
	QSamples map[int][]float64
	Q        map[int]float64
	Refine   RefinePlan
}

type ExecConfig struct {
	DetectionPath     string
	FramePath         string
	TrackOutput       string
	FilterOutput      string
	UncertaintyOutput string
	RefineOutput      string
	OutPath           string
}
