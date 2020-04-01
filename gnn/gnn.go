package gnn

import (
	"../miris"

	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
)

type GNN struct {
	cmd *exec.Cmd
	stdin io.WriteCloser
	rd *bufio.Reader
	detections [][]miris.Detection
}

func NewGNN(modelPath string, detectionPath string, framePath string, frameScale int) *GNN {
	var detections [][]miris.Detection
	miris.ReadJSON(detectionPath, &detections)

	cmd := exec.Command("python", "models/gnn/wrapper.py", modelPath, detectionPath, framePath, strconv.Itoa(frameScale))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(err)
	}
	err = cmd.Start()
	if err != nil {
		panic(err)
	}

	go miris.LogStderr("gnn", stderr)
	rd := bufio.NewReader(stdout)
	return &GNN{cmd, stdin, rd, detections}
}

func (gnn *GNN) NumFrames() int {
	return len(gnn.detections)
}

func (gnn *GNN) Infer(idx1 int, idx2 int) [][]float64 {
	input := fmt.Sprintf("%d %d\n", idx1, idx2)
	if _, err := gnn.stdin.Write([]byte(input)); err != nil {
		panic(err)
	}
	line, err := gnn.rd.ReadString('\n')
	if err != nil {
		panic(err)
	}
	var mat [][]float64
	if err := json.Unmarshal([]byte(line), &mat); err != nil {
		panic(err)
	}
	return mat
}

func (gnn *GNN) Close() {
	gnn.stdin.Close()
	gnn.cmd.Wait()
}
