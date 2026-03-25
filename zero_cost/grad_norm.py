# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch

import torch.nn.functional as F

import copy

from zero_cost.zero_utils import measure


@measure('grad_norm', bn=True)
def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    
    # Forward pass
    outputs = net(inputs)
    
    # Handle tuple outputs (e.g. from auxiliary heads)
    if isinstance(outputs, tuple):
        output = outputs[0]
    else:
        output = outputs
        
    loss = loss_fn(output, targets)
    loss.backward()

    grad_norm_arr = []
    
    # Iterate directly over parameters instead of modules to avoid duplicates and ensure valid grads
    for param in net.parameters():
        if param.grad is not None:
            # Calculate L2 norm of the gradient for this parameter tensor
            # view(-1) flattens the tensor to 1D, ensuring norm calculation is correct
            grad_norm = torch.norm(param.grad.view(-1))
            grad_norm_arr.append(grad_norm)
            
    return grad_norm_arr
