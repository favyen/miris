package miris

import (
	"bufio"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"strings"
)

func LogStderr(prefix string, stderr io.ReadCloser) {
	rd := bufio.NewReader(stderr)
	for {
		line, err := rd.ReadString('\n')
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
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
