package miris

import (
	"bufio"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"os/exec"
	"strings"
)

func LogStderr(prefix string, stderr io.ReadCloser) {
	rd := bufio.NewReader(stderr)
	for {
		line, err := rd.ReadString('\n')
		if err != nil {
			break
		}
		log.Printf("[%s] %s", prefix, strings.TrimSpace(line))
	}
}

func ReadJSON(fname string, x interface{}) {
	bytes, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(bytes, x); err != nil {
		panic(err)
	}
}

func WriteJSON(fname string, x interface{}) {
	bytes, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(fname, bytes, 0644); err != nil {
		panic(err)
	}
}

func Command(prefix string, command string, args ...string) (*exec.Cmd, io.WriteCloser, io.ReadCloser) {
	cmd := exec.Command(command, args...)
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
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	go LogStderr(prefix, stderr)
	return cmd, stdin, stdout
}
