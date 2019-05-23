#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

//b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
//b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
//b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
//b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)
//
//dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
//dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
//intersections = dx * dy
//
//areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
//areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
//unions = (areas1 + areas2.t()) - intersections
//
//return intersections / unions

//boxes1, boxes2
torch::Tensor bbox_ious(
    torch::Tensor boxes1_xy,
    torch::Tensor boxes1_wh,
    torch::Tensor boxes2_xy,
    torch::Tensor boxes2_wh
    ) {
    auto b1x1y1 = boxes1_xy-boxes1_wh/2;
    auto b1x2y2 = boxes1_xy+boxes1_wh/2;

    auto b1x1y1_split = b1x1y1.split(1, 1);
    auto b1x1=b1x1y1_split[0];
    auto b1y1=b1x1y1_split[1];

    auto b1x2y2_split = b1x2y2.split(1, 1);
    auto b1x2=b1x2y2_split[0];
    auto b1y2=b1x2y2_split[1];

    auto b2x1y1 = boxes1_xy-boxes1_wh/2;
    auto b2x2y2 = boxes1_xy+boxes1_wh/2;

    auto b2x1y1_split = b2x1y1.split(1, 1);
    auto b2x1=b2x1y1_split[0];
    auto b2y1=b2x1y1_split[1];

    auto b2x2y2_split = b2x2y2.split(1, 1);
    auto b2x2=b2x2y2_split[0];
    auto b2y2=b2x2y2_split[1];


    auto dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(0);
    auto dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(0);
    auto intersections = dx * dy;

    auto areas1 = (b1x2 - b1x1) * (b1y2 - b1y1);
    auto areas2 = (b2x2 - b2x1) * (b2y2 - b2y1);
    auto unions = (areas1 + areas2.t()) - intersections;
    return intersections / unions;

//    b1x1 = boxes1[:, :1]
//  auto X = torch::cat({old_h, input}, /*dim=*/1);
//
//  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
//  auto gates = gate_weights.chunk(3, /*dim=*/1);
//
//  auto input_gate = torch::sigmoid(gates[0]);
//  auto output_gate = torch::sigmoid(gates[1]);
//  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);
//
//  auto new_cell = old_cell + candidate_cell * input_gate;
//  auto new_h = torch::tanh(new_cell) * output_gate;

//  return {boxes1, boxes2};
//  return {b1x1y1};
//    return b1x1y1_split;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
  m.def("bbox_ious", &bbox_ious, "bbox_ious");
}
