require "torch"
require "torchvision"
require "nn"


model = torch.load("model.ckpt")

input = torch.Tensor{{1, 2}}

output = model:forward(input)

print(output)
