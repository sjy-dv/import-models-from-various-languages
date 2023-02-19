#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  GraphDef graph_def;
  Status status = ReadBinaryProto(Env::Default(), "model.ckpt", &graph_def);
  if (!status.ok()) {
    std::cout << "Error reading graph definition from file: " << status.ToString() << std::endl;
    return 1;
  }

  ClientSession session(root);
  status = session.Create(graph_def);
  if (!status.ok()) {
    std::cout << "Error creating session: " << status.ToString() << std::endl;
    return 1;
  }

  
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "input", tensorflow::Tensor(DT_FLOAT, TensorShape({1, 2})) }
  };
  std::vector<tensorflow::Tensor> outputs;

  auto input_tensor = inputs[0].second.tensor<float, 2>();
  input_tensor(0, 0) = 1.0;
  input_tensor(0, 1) = 2.0;

  status = session.Run(inputs, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << "Error running session: " << status.ToString() << std::endl;
    return 1;
  }

  auto output_tensor = outputs[0].tensor<float, 2>();
  std::cout << output_tensor(0, 0) << std::endl;

  return 0;
}
