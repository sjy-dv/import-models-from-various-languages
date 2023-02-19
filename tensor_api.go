package main

import (
	"fmt"
)

func main() {

	graph, err := tensorflow.LoadGraph("model.ckpt", "")
	if err != nil {
		panic(err)
	}
	defer graph.Close()

	inputTensor := graph.Operation("input").Output(0)
	outputTensor := graph.Operation("output").Output(0)

	inputArray := []float32{1.0, 2.0}
	inputTensorShape := []int64{1, 2}
	inputTensor, err := tensorflow.NewTensor(inputArray, inputTensorShape)
	if err != nil {
		panic(err)
	}

	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	defer session.Close()
	output, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			inputTensor.Operation().Output(0): inputTensor,
		},
		[]tensorflow.Output{outputTensor},
		nil,
	)
	if err != nil {
		panic(err)
	}

	outputData := output[0].Value().([][]float32)
	fmt.Println(outputData)
}
