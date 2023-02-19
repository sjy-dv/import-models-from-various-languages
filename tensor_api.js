const tf = require("@tensorflow/tfjs-node-gpu");

const graph = tf.GraphModel.fromCheckpoint("model.ckpt");

const inputTensor = graph.inputs[0];
const outputTensor = graph.outputs[0];

const inputArray = tf.tensor2d([[1.0, 2.0]]);
const input = inputArray.expandDims(0);

const output = graph.predict(input);

output.print();
