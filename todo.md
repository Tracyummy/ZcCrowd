
- todo
```python
class BinarizedF(Function):
  @staticmethod
  def forward(ctx, input, threshold):
    ctx.save_for_backward(input,threshold)
    a = torch.ones_like(input).cuda()
    b = torch.zeros_like(input).cuda()
    output = torch.where(input>=threshold,a,b)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    # print('grad_output',grad_output)
    input,threshold = ctx.saved_tensors
    grad_input = grad_weight  = None

    if ctx.needs_input_grad[0]:
      grad_input= 0.2*grad_output
    if ctx.needs_input_grad[1]:
      grad_weight = -grad_output
    return grad_input, grad_weight
```

```python
Binar_map = BinarizedF.apply(pred_map, threshold)
```

```python
output = torch.where(input>=threshold,a,b)
```

```python

```

