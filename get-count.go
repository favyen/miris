package main

import (
	"./miris"
	"./predicate"

	"fmt"
	"os"
)

func main() {
	predName := os.Args[1]
	trackFname := os.Args[2]

	predFunc := predicate.GetPredicate(predName)
	var detections [][]miris.Detection
	miris.ReadJSON(trackFname, &detections)
	tracks := miris.GetTracks(detections)
	var count int = 0
	for _, track := range tracks {
		if predFunc([][]miris.Detection{track}) {
			count++
		}
	}
	fmt.Println(predName, count)
}
