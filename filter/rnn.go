package filter

import (
	"../miris"

	"bufio"
	"encoding/json"
	"io"
	"os/exec"
)

func init() {
	FilterMap["rnn"] = MakeRNNFilter
}

type RNNFilter struct {
	cmd *exec.Cmd
	stdin io.WriteCloser
	rd *bufio.Reader
}

func MakeRNNFilter(freq int, tracks [][]miris.Detection, labels []bool, cfg map[string]string) Filter {
	cmd := exec.Command("python", "models/rnn/wrapper.py", "filter", cfg["model_path"])
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

	go miris.LogStderr("filter-rnn", stderr)
	rd := bufio.NewReader(stdout)
	return RNNFilter{cmd, stdin, rd}
}

func (f RNNFilter) Predict(tracks [][]miris.Detection) []float64 {
	bytes, err := json.Marshal(tracks)
	if err != nil {
		panic(err)
	}
	if _, err := f.stdin.Write([]byte(string(bytes)+"\n")); err != nil {
		panic(err)
	}
	line, err := f.rd.ReadString('\n')
	if err != nil {
		panic(err)
	}
	var scores []float64
	if err := json.Unmarshal([]byte(line), &scores); err != nil {
		panic(err)
	}
	return scores
}

func (f RNNFilter) Close() {
	f.stdin.Close()
	f.cmd.Wait()
}
