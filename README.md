
================================================================================
FILE: issue_links.csv
================================================================================
Link,# Bugs
https://github.com/pytorch/pytorch/issues/51933,1
https://github.com/pytorch/pytorch/issues/53466,1
https://github.com/pytorch/pytorch/issues/54319,1
https://github.com/pytorch/pytorch/issues/54320,1
https://github.com/pytorch/pytorch/issues/55356,1
https://github.com/pytorch/pytorch/issues/55360,1
https://github.com/pytorch/pytorch/issues/56224,1
https://github.com/pytorch/pytorch/issues/56263,1
https://github.com/pytorch/pytorch/issues/49890,6
https://github.com/pytorch/pytorch/issues/55359,1
https://github.com/pytorch/pytorch/issues/55381,1
https://github.com/pytorch/pytorch/issues/56246,1
https://github.com/pytorch/pytorch/issues/56328,1
https://github.com/pytorch/pytorch/issues/56330,1
https://github.com/pytorch/pytorch/issues/65447,1
https://github.com/pytorch/pytorch/issues/65520,1
https://github.com/pytorch/pytorch/issues/66751,1
https://github.com/pytorch/pytorch/issues/66872,1
https://github.com/pytorch/pytorch/issues/66750,1
https://github.com/pytorch/pytorch/issues/65399,1
https://github.com/pytorch/pytorch/issues/68727,1
https://github.com/pytorch/pytorch/issues/66868,1
https://github.com/pytorch/pytorch/issues/70489,1
https://github.com/tensorflow/tensorflow/issues/48470,4
https://github.com/tensorflow/tensorflow/issues/48470,3
https://github.com/tensorflow/tensorflow/issues/48467,1
https://github.com/tensorflow/tensorflow/issues/48469,1
https://github.com/tensorflow/tensorflow/issues/48481,1
https://github.com/tensorflow/tensorflow/issues/48477,1
https://github.com/tensorflow/tensorflow/issues/48466,1
https://github.com/tensorflow/tensorflow/issues/48589,1
https://github.com/tensorflow/tensorflow/issues/51618,1
https://github.com/tensorflow/tensorflow/issues/51624,1
https://github.com/tensorflow/tensorflow/issues/51625,1
https://github.com/tensorflow/tensorflow/issues/51936,2
https://github.com/tensorflow/tensorflow/issues/52063,1
https://github.com/tensorflow/tensorflow/issues/53300,1
https://github.com/tensorflow/tensorflow/issues/51908,1
Total,49


================================================================================
FILE: README.md
================================================================================
# FreeFuzz

This is the artifact of the research paper, "Free Lunch for Testing: Fuzzing Deep-Learning Libraries from Open Source", at ICSE 2022.

## About

FreeFuzz is the first approach to fuzzing DL libraries via mining from open source. It collects code/models from three different sources: 1) code snippets from the library documentation, 2) library developer tests, and 3) DL models in the wild. Then, FreeFuzz automatically runs all the collected code/models with instrumentation to collect the dynamic information for each covered API. Lastly, FreeFuzz will leverage the traced dynamic information to perform fuzz testing for each covered API.

This is the FreeFuzz's implementation for testing PyTorch and TensorFlow.

## Getting Started

### 1. Requirements

1. Our testing framework leverages [MongoDB](https://www.mongodb.com/) so you should [install and run MongoDB](https://docs.mongodb.com/manual/installation/) first.
	- Run the command `ulimit -n 64000` to adjust the limit that the system resources a process may use. You can see this [document](https://docs.mongodb.com/manual/reference/ulimit/) for more details.
2. You should check our dependent python libraries in `requirements.txt` and run `pip install -r requirements.txt` to install them
3. Python version >= 3.8.0 (It must support f-string.)

### 2. Setting Up with Dataset

#### Using Our Dataset

Run the following command to load the database.

```shell
mongorestore dump/
```

#### Collecting Data by Yourself

1. Go to `src/instrumentation/{torch, tensorflow}` to see how to intrument the dynamic information and add them into the database
2. After adding invocation data, you should run the following command to preprocess the data for PyTorch

```shell
cd src && python preprocess/process_data.py torch
```

or for TensorFlow
```shell
cd src && python preprocess/process_data.py tf
```

### 3. Configuration

There are some hyper-parameters in FreeFuzz and they could be easily configured as follows.

In `src/config/demo.conf`:

1. MongoDB database configuration.

```conf
[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = freefuzz-torch
# name of tensorflow database
tf_database = freefuzz-tf
```

2. Output directory configuration.

```conf
[output]
# output directory for pytorch
torch_output = torch-output
# output directory for tensorflow
tf_output = tf-output
```

3. Oracle configuration.

```conf
[oracle]
# enable crash oracle
enable_crash = true
# enable cuda oracle
enable_cuda = true
# enable precision oracle
enable_precision = true
# float difference bound: if |a-b| > bound, a is different than b
float_difference_bound = 1e-5
# max time bound: if time(low_precision) > bound * time(high_precision),
# it will be considered as a potential bug
max_time_bound = 10
# only consider the call with time(call) > time_thresold
time_thresold = 1e-3
```

4. Mutation stratgy configuration.

```conf
[mutation]
enable_value_mutation = true
enable_type_mutation = true
enable_db_mutation = true
# the number of times each api is executed
each_api_run_times = 1000
```

### 4. Start

After finishing above steps, run the following command to start FreeFuzz to test PyTorch

```shell
cd src && python FreeFuzz.py --conf demo_torch.conf
```

Or run this command to test TensorFlow

```shell
cd src && python FreeFuzz.py --conf demo_tf.conf
```

To run the full experiment, run the following command
```shell
cd src && python FreeFuzz.py --conf expr.conf
```
If you want to use another configuration file, you can put it in `src/config`.

Note that you should specify the configuration file you want to use.

## Notes

1. Some APIs will be skipped since they may crash the program. You can set what you want to skip in the file `src/config/skip_torch.txt` or `src/config/skip_tf`.
2. For the details of three mutation strategies, please refer to our paper.


================================================================================
FILE: requirements.txt
================================================================================
torch
tensorflow
pymongo
numpy
configparser
textdistance


================================================================================
FILE: data\torch_APIdef.txt
================================================================================
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
torch.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
torch.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
torch.nn.functional.max_pool1d(*args, **kwargs)
torch.nn.functional.max_pool2d(*args, **kwargs)
torch.nn.functional.max_pool3d(*args, **kwargs)
torch.nn.functional.max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
torch.nn.functional.max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
torch.nn.functional.lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
torch.nn.functional.adaptive_max_pool1d(*args, **kwargs)
torch.nn.functional.adaptive_max_pool2d(*args, **kwargs)
torch.nn.functional.adaptive_max_pool3d(*args, **kwargs)
torch.nn.functional.adaptive_avg_pool1d(input, output_size)
torch.nn.functional.adaptive_avg_pool2d(input, output_size)
torch.nn.functional.adaptive_avg_pool3d(input, output_size)
torch.nn.functional.threshold(input, threshold, value, inplace=False)
torch.nn.functional.threshold_(input, threshold, value)
torch.nn.functional.relu(input, inplace=False)
torch.nn.functional.relu_(input)
torch.nn.functional.hardtanh(input, min_val=-1., max_val=1., inplace=False)
torch.nn.functional.hardtanh_(input, min_val=-1., max_val=1.)
torch.nn.functional.hardswish(input, inplace=False)
torch.nn.functional.relu6(input, inplace=False)
torch.nn.functional.elu(input, alpha=1.0, inplace=False)
torch.nn.functional.elu_(input, alpha=1.)
torch.nn.functional.selu(input, inplace=False)
torch.nn.functional.celu(input, alpha=1., inplace=False)
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
torch.nn.functional.leaky_relu_(input, negative_slope=0.01)
torch.nn.functional.prelu(input, weight)
torch.nn.functional.rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False)
torch.nn.functional.rrelu_(input, lower=1./8, upper=1./3, training=False)
torch.nn.functional.glu(input, dim=-1)
torch.nn.functional.gelu(input)
torch.nn.functional.logsigmoid(input)
torch.nn.functional.hardshrink(input, lambd=0.5)
torch.nn.functional.tanhshrink(input)
torch.nn.functional.softsign(input)
torch.nn.functional.softplus(input, beta=1, threshold=20)
torch.nn.functional.softmin(input, dim=None, _stacklevel=3, dtype=None)
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
torch.nn.functional.softshrink(input, lambd=0.5)
torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1)
torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
torch.nn.functional.tanh(input)
torch.nn.functional.sigmoid(input)
torch.nn.functional.hardsigmoid(input, inplace=False)
torch.nn.functional.silu(input, inplace=False)
torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
torch.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)
torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
torch.nn.functional.local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0)
torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
torch.nn.functional.linear(input, weight, bias=None)
torch.nn.functional.bilinear(input1, input2, weight, bias=None)
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
torch.nn.functional.alpha_dropout(input, p=0.5, training=False, inplace=False)
torch.nn.functional.feature_alpha_dropout(input, p=0.5, training=False, inplace=False)
torch.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False)
torch.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False)
torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
torch.nn.functional.embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None)
torch.nn.functional.one_hot(tensor, num_classes=-1)
torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False)
torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)
torch.nn.functional.pdist(input, p=2)
torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
torch.nn.functional.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
torch.nn.functional.cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)
torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.multilabel_soft_margin_loss(input, target, weight=None, size_average=None)
torch.nn.functional.multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.functional.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0)
torch.nn.functional.soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, *, distance_function=None, margin=1.0, swap=False, reduction='mean')
torch.nn.functional.pixel_shuffle(input, upscale_factor)
torch.nn.functional.pixel_unshuffle(input, downscale_factor)
torch.nn.functional.pad(input, pad, mode='constant', value=0)
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
torch.nn.functional.upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
torch.nn.functional.upsample_nearest(input, size=None, scale_factor=None)
torch.nn.functional.upsample_bilinear(input, size=None, scale_factor=None)
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
torch.nn.functional.affine_grid(theta, size, align_corners=None)
torch.is_tensor(obj)
torch.is_storage(obj)
torch.is_complex(input)
torch.is_floating_point(input)
torch.is_nonzero(input)
torch.set_default_dtype(d)
torch.get_default_dtype()
torch.set_default_tensor_type(t)
torch.numel(input)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
torch.set_flush_denormal(mode)
torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format)
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False)
torch.as_tensor(data, dtype=None, device=None)
torch.as_strided(input, size, stride, storage_offset=0)
torch.from_numpy(ndarray)
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False)
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.full_like(input, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
torch.quantize_per_tensor(input, scale, zero_point, dtype)
torch.quantize_per_channel(input, scales, zero_points, axis, dtype)
torch.dequantize(tensor)
torch.dequantize(tensors)
torch.complex(real, imag, *, out=None)
torch.imag(input)
torch.polar(abs, angle, *, out=None)
torch.angle(input, *, out=None)
torch.heaviside(input, values, *, out=None)
torch.cat(tensors, dim=0, *, out=None)
torch.chunk(input, chunks, dim=0)
torch.column_stack(tensors, *, out=None)
torch.dstack(tensors, *, out=None)
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
torch.hstack(tensors, *, out=None)
torch.index_select(input, dim, index, *, out=None)
torch.masked_select(input, mask, *, out=None)
torch.movedim(input, source, destination)
torch.moveaxis(input, source, destination)
torch.narrow(input, dim, start, length)
torch.nonzero(input, *, out=None, as_tuple=False)
torch.reshape(input, shape)
torch.row_stack(tensors, *, out=None)
torch.vstack(tensors, *, out=None)
torch.scatter(input, dim, index, src)
torch.scatter_add(input, dim, index, src)
torch.split(tensor, split_size_or_sections, dim=0)
torch.squeeze(input, dim=None, *, out=None)
torch.stack(tensors, dim=0, *, out=None)
torch.swapaxes(input, axis0, axis1)
torch.transpose(input, dim0, dim1)
torch.swapdims(input, dim0, dim1)
torch.t(input)
torch.take(input, index)
torch.tensor_split(input, indices_or_sections, dim=0)
torch.tile(input, reps)
torch.unbind(input, dim=0)
torch.unsqueeze(input, dim)
torch.where(condition, x, y)
torch.where(condition)
torch.Generator(device='cpu')
torch.seed()
torch.manual_seed(seed)
torch.initial_seed()
torch.get_rng_state()
torch.set_rng_state(new_state)
torch.bernoulli(input, *, generator=None, out=None)
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
torch.normal(mean, std, *, generator=None, out=None)
torch.normal(mean=0.0, std, *, out=None)
torch.normal(mean, std=1.0, *, out=None)
torch.normal(mean, std, size, *, out=None)
torch.poisson(input, generator=None)
torch.quasirandom.SobolEngine(dimension, scramble=False, seed=None)
torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)
torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)
torch.get_num_threads()
torch.set_num_threads(int)
torch.get_num_interop_threads()
torch.set_num_interop_threads(int)
torch.set_grad_enabled(mode)
torch.abs(input, *, out=None)
torch.absolute(input, *, out=None)
torch.acos(input, *, out=None)
torch.arccos(input, *, out=None)
torch.acosh(input, *, out=None)
torch.arccosh(input, *, out=None)
torch.add(input, other, *, out=None)
torch.add(input, other, *, alpha=1, out=None)
torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
torch.asin(input, *, out=None)
torch.arcsin(input, *, out=None)
torch.asinh(input, *, out=None)
torch.arcsinh(input, *, out=None)
torch.atan(input, *, out=None)
torch.arctan(input, *, out=None)
torch.atanh(input, *, out=None)
torch.arctanh(input, *, out=None)
torch.atan2(input, other, *, out=None)
torch.bitwise_not(input, *, out=None)
torch.bitwise_and(input, other, *, out=None)
torch.bitwise_or(input, other, *, out=None)
torch.bitwise_xor(input, other, *, out=None)
torch.ceil(input, *, out=None)
torch.clamp(input, min=None, max=None, *, out=None)
torch.max(input)
torch.max(input, dim, keepdim=False, *, out=None)
torch.max(input, other, *, out=None)
torch.clip(input, min=None, max=None, *, out=None)
torch.conj(input, *, out=None)
torch.copysign(input, other, *, out=None)
torch.cos(input, *, out=None)
torch.cosh(input, *, out=None)
torch.deg2rad(input, *, out=None)
torch.div(input, other, *, rounding_mode=None, out=None)
torch.divide(input, other, *, rounding_mode=None, out=None)
torch.digamma(input, *, out=None)
torch.erf(input, *, out=None)
torch.erfc(input, *, out=None)
torch.erfinv(input, *, out=None)
torch.exp(input, *, out=None)
torch.exp2(input, *, out=None)
torch.expm1(input, *, out=None)
torch.fake_quantize_per_channel_affine(input, scale, zero_point, quant_min, quant_max)
torch.fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)
torch.fix(input, *, out=None)
torch.trunc(input, *, out=None)
torch.float_power(input, exponent, *, out=None)
torch.floor(input, *, out=None)
torch.floor_divide(input, other, *, out=None)
torch.fmod(input, other, *, out=None)
torch.frac(input, *, out=None)
torch.ldexp(input, other, *, out=None)
torch.lerp(input, end, weight, *, out=None)
torch.lgamma(input, *, out=None)
torch.log(input, *, out=None)
torch.log10(input, *, out=None)
torch.log1p(input, *, out=None)
torch.log2(input, *, out=None)
torch.logaddexp(input, other, *, out=None)
torch.logaddexp2(input, other, *, out=None)
torch.logical_and(input, other, *, out=None)
torch.logical_not(input, *, out=None)
torch.logical_or(input, other, *, out=None)
torch.logical_xor(input, other, *, out=None)
torch.logit(input, eps=None, *, out=None)
torch.hypot(input, other, *, out=None)
torch.i0(input, *, out=None)
torch.igamma(input, other, *, out=None)
torch.igammac(input, other, *, out=None)
torch.mul(input, other, *, out=None)
torch.mul(input, other, *, out=None)
torch.multiply(input, other, *, out=None)
torch.mvlgamma(input, p)
torch.nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None)
torch.neg(input, *, out=None)
torch.negative(input, *, out=None)
torch.nextafter(input, other, *, out=None)
torch.polygamma(n, input, *, out=None)
torch.pow(input, exponent, *, out=None)
torch.pow(self, exponent, *, out=None)
torch.rad2deg(input, *, out=None)
torch.real(input)
torch.reciprocal(input, *, out=None)
torch.remainder(input, other, *, out=None)
torch.round(input, *, out=None)
torch.rsqrt(input, *, out=None)
torch.sigmoid(input, *, out=None)
torch.sign(input, *, out=None)
torch.sgn(input, *, out=None)
torch.signbit(input, *, out=None)
torch.sin(input, *, out=None)
torch.sinc(input, *, out=None)
torch.sinh(input, *, out=None)
torch.sqrt(input, *, out=None)
torch.square(input, *, out=None)
torch.sub(input, other, *, alpha=1, out=None)
torch.subtract(input, other, *, alpha=1, out=None)
torch.tan(input, *, out=None)
torch.tanh(input, *, out=None)
torch.true_divide(dividend, divisor, *, out)
torch.xlogy(input, other, *, out=None)
torch.argmax(input)
torch.argmax(input, dim, keepdim=False)
torch.argmin(input, dim=None, keepdim=False)
torch.amax(input, dim, keepdim=False, *, out=None)
torch.amin(input, dim, keepdim=False, *, out=None)
torch.all(input)
torch.all(input, dim, keepdim=False, *, out=None)
torch.any(input)
torch.any(input, dim, keepdim=False, *, out=None)
torch.min(input)
torch.min(input, dim, keepdim=False, *, out=None)
torch.min(input, other, *, out=None)
torch.dist(input, other, p=2)
torch.logsumexp(input, dim, keepdim=False, *, out=None)
torch.mean(input)
torch.mean(input, dim, keepdim=False, *, out=None)
torch.median(input)
torch.median(input, dim=-1, keepdim=False, *, out=None)
torch.nanmedian(input)
torch.nanmedian(input, dim=-1, keepdim=False, *, out=None)
torch.mode(input, dim=-1, keepdim=False, *, out=None)
torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
torch.nansum(input, *, dtype=None)
torch.nansum(input, dim, keepdim=False, *, dtype=None)
torch.prod(input, *, dtype=None)
torch.prod(input, dim, keepdim=False, *, dtype=None)
torch.quantile(input, q, dim=None, keepdim=False, *, out=None)
torch.nanquantile(input, q, dim=None, keepdim=False, *, out=None)
torch.std(input, dim, unbiased, keepdim=False, *, out=None)
torch.std(input, unbiased)
torch.std_mean(input, dim, unbiased, keepdim=False, *, out=None)
torch.std_mean(input, unbiased)
torch.sum(input, *, dtype=None)
torch.sum(input, dim, keepdim=False, *, dtype=None)
torch.unique(*args, **kwargs)
torch.unique_consecutive(*args, **kwargs)
torch.var(input, dim, unbiased, keepdim=False, *, out=None)
torch.var(input, unbiased)
torch.var_mean(input, dim, unbiased, keepdim=False, *, out=None)
torch.var_mean(input, unbiased)
torch.count_nonzero(input, dim=None)
torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)
torch.argsort(input, dim=-1, descending=False)
torch.eq(input, other, *, out=None)
torch.equal(input, other)
torch.ge(input, other, *, out=None)
torch.greater_equal(input, other, *, out=None)
torch.gt(input, other, *, out=None)
torch.greater(input, other, *, out=None)
torch.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)
torch.isfinite(input)
torch.isinf(input)
torch.isposinf(input, *, out=None)
torch.isneginf(input, *, out=None)
torch.isnan(input)
torch.isreal(input)
torch.kthvalue(input, k, dim=None, keepdim=False, *, out=None)
torch.le(input, other, *, out=None)
torch.less_equal(input, other, *, out=None)
torch.lt(input, other, *, out=None)
torch.less(input, other, *, out=None)
torch.maximum(input, other, *, out=None)
torch.minimum(input, other, *, out=None)
torch.fmax(input, other, *, out=None)
torch.fmin(input, other, *, out=None)
torch.ne(input, other, *, out=None)
torch.not_equal(input, other, *, out=None)
torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
torch.msort(input, *, out=None)
torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None)
torch.istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False)
torch.bartlett_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.blackman_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.atleast_1d(*tensors)
torch.atleast_2d(*tensors)
torch.atleast_3d(*tensors)
torch.bincount(input, weights=None, minlength=0)
torch.block_diag(*tensors)
torch.broadcast_tensors(*tensors)
torch.broadcast_to(input, shape)
torch.broadcast_shapes(*shapes)
torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None)
torch.cartesian_prod(*tensors)
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
torch.clone(input, *, memory_format=torch.preserve_format)
torch.combinations(input, r=2, with_replacement=False)
torch.cross(input, other, dim=None, *, out=None)
torch.cummax(input, dim, *, out=None)
torch.cummin(input, dim, *, out=None)
torch.cumprod(input, dim, *, dtype=None, out=None)
torch.cumsum(input, dim, *, dtype=None, out=None)
torch.diag(input, diagonal=0, *, out=None)
torch.diag_embed(input, offset=0, dim1=-2, dim2=-1)
torch.diagflat(input, offset=0)
torch.diagonal(input, offset=0, dim1=0, dim2=1)
torch.diff(input, n=1, dim=-1, prepend=None, append=None)
torch.einsum(equation, *operands)
torch.flatten(input, start_dim=0, end_dim=-1)
torch.flip(input, dims)
torch.fliplr(input)
torch.flipud(input)
torch.kron(input, other, *, out=None)
torch.rot90(input, k, dims)
torch.gcd(input, other, *, out=None)
torch.histc(input, bins=100, min=0, max=0, *, out=None)
torch.meshgrid(*tensors)
torch.lcm(input, other, *, out=None)
torch.logcumsumexp(input, dim, *, out=None)
torch.ravel(input)
torch.renorm(input, p, dim, maxnorm, *, out=None)
torch.repeat_interleave(input, repeats, dim=None)
torch.repeat_interleave(repeats)
torch.roll(input, shifts, dims=None)
torch.searchsorted(sorted_sequence, values, *, out_int32=False, right=False, out=None)
torch.tensordot(a, b, dims=2, out=None)
torch.trace(input)
torch.tril(input, diagonal=0, *, out=None)
torch.tril_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided)
torch.triu(input, diagonal=0, *, out=None)
torch.triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided)
torch.vander(x, N=None, increasing=False)
torch.view_as_real(input)
torch.view_as_complex(input)
torch.addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
torch.addmv(input, mat, vec, *, beta=1, alpha=1, out=None)
torch.addr(input, vec1, vec2, *, beta=1, alpha=1, out=None)
torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
torch.bmm(input, mat2, *, deterministic=False, out=None)
torch.chain_matmul(*matrices, out=None)
torch.cholesky(input, upper=False, *, out=None)
torch.cholesky_inverse(input, upper=False, *, out=None)
torch.cholesky_solve(input, input2, upper=False, *, out=None)
torch.dot(input, other, *, out=None)
torch.eig(input, eigenvectors=False, *, out=None)
torch.geqrf(input, *, out=None)
torch.ger(input, vec2, *, out=None)
torch.outer(input, vec2, *, out=None)
torch.inner(input, other, *, out=None)
torch.inverse(input, *, out=None)
torch.det(input)
torch.logdet(input)
torch.slogdet(input)
torch.lstsq(input, A, *, out=None)
torch.lu(*args, **kwargs)
torch.lu_solve(b, LU_data, LU_pivots, *, out=None)
torch.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None)
torch.matmul(input, other, *, out=None)
torch.matrix_power(input, n, *, out=None)
torch.matrix_rank(input, tol=None, symmetric=False, *, out=None)
torch.matrix_exp(input)
torch.mm(input, mat2, *, out=None)
torch.mv(input, vec, *, out=None)
torch.orgqr(input, tau)
torch.ormqr(input, tau, other, left=True, transpose=False, *, out=None)
torch.pinverse(input, rcond=1e-15)
torch.qr(input, some=True, *, out=None)
torch.solve(input, A, *, out=None)
torch.svd(input, some=True, compute_uv=True, *, out=None)
torch.svd_lowrank(A, q=6, niter=2, M=None)
torch.pca_lowrank(A, q=None, center=True, niter=2)
torch.symeig(input, eigenvectors=False, upper=True, *, out=None)
torch.lobpcg(A, k=None, B=None, X=None, n=None, iK=None, niter=None, tol=None, largest=None, method=None, tracker=None, ortho_iparams=None, ortho_fparams=None, ortho_bparams=None)
torch.trapz(y, x, *, dim=-1)
torch.trapz(y, *, dx=1, dim=-1)
torch.triangular_solve(b, A, upper=True, transpose=False, unitriangular=False)
torch.vdot(input, other, *, out=None)
torch.compiled_with_cxx11_abi()
torch.result_type(tensor1, tensor2)
torch.can_cast(from, to)
torch.promote_types(type1, type2)
torch.use_deterministic_algorithms(mode)
torch.are_deterministic_algorithms_enabled()
torch._assert(condition, message)
torch.nn.Sequential(*args)
torch.nn.ModuleList(modules=None)
torch.nn.ModuleDict(modules=None)
torch.nn.ParameterList(parameters=None)
torch.nn.ParameterDict(parameters=None)
torch.nn.modules.module.register_module_forward_pre_hook(hook)
torch.nn.modules.module.register_module_forward_hook(hook)
torch.nn.modules.module.register_module_backward_hook(hook)
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConv1d(out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConv2d(out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConv3d(out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConvTranspose1d(out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConvTranspose2d(out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.LazyConvTranspose3d(out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
torch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
torch.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
torch.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
torch.nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)
torch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)
torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
torch.nn.AdaptiveMaxPool3d(output_size, return_indices=False)
torch.nn.AdaptiveAvgPool1d(output_size)
torch.nn.AdaptiveAvgPool2d(output_size)
torch.nn.AdaptiveAvgPool3d(output_size)
torch.nn.ReflectionPad1d(padding)
torch.nn.ReflectionPad2d(padding)
torch.nn.ReplicationPad1d(padding)
torch.nn.ReplicationPad2d(padding)
torch.nn.ReplicationPad3d(padding)
torch.nn.ZeroPad2d(padding)
torch.nn.ConstantPad1d(padding, value)
torch.nn.ConstantPad2d(padding, value)
torch.nn.ConstantPad3d(padding, value)
torch.nn.ELU(alpha=1.0, inplace=False)
torch.nn.Hardshrink(lambd=0.5)
torch.nn.Hardsigmoid(inplace=False)
torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None)
torch.nn.Hardswish(inplace=False)
torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
torch.nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)
torch.nn.ReLU(inplace=False)
torch.nn.ReLU6(inplace=False)
torch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
torch.nn.SELU(inplace=False)
torch.nn.CELU(alpha=1.0, inplace=False)
torch.nn.SiLU(inplace=False)
torch.nn.Softplus(beta=1, threshold=20)
torch.nn.Softshrink(lambd=0.5)
torch.nn.Threshold(threshold, value, inplace=False)
torch.nn.Softmin(dim=None)
torch.nn.Softmax(dim=None)
torch.nn.LogSoftmax(dim=None)
torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, device=None, dtype=None)
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
torch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, device=None, dtype=None)
torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)
torch.nn.RNNBase(mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
torch.nn.RNN(*args, **kwargs)
torch.nn.LSTM(*args, **kwargs)
torch.nn.GRU(*args, **kwargs)
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
torch.nn.LSTMCell(input_size, hidden_size, bias=True, device=None, dtype=None)
torch.nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
torch.nn.Identity(*args, **kwargs)
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
torch.nn.Bilinear(in1_features, in2_features, out_features, bias=True, device=None, dtype=None)
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)
torch.nn.Dropout(p=0.5, inplace=False)
torch.nn.Dropout2d(p=0.5, inplace=False)
torch.nn.Dropout3d(p=0.5, inplace=False)
torch.nn.AlphaDropout(p=0.5, inplace=False)
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
torch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None, include_last_offset=False, padding_idx=None, device=None, dtype=None)
torch.nn.CosineSimilarity(dim=1, eps=1e-08)
torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
torch.nn.GaussianNLLLoss(*, full=False, eps=1e-06, reduction='mean')
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
torch.nn.TripletMarginWithDistanceLoss(*, distance_function=None, margin=1.0, swap=False, reduction='mean')
torch.nn.PixelShuffle(upscale_factor)
torch.nn.PixelUnshuffle(downscale_factor)
torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)
torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)
torch.nn.ChannelShuffle(groups)
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=25, find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False)
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)
torch.nn.utils.clip_grad_value_(parameters, clip_value)
torch.nn.utils.parameters_to_vector(parameters)
torch.nn.utils.vector_to_parameters(vec, parameters)
torch.nn.utils.prune.PruningContainer(*args)
torch.nn.utils.prune.RandomUnstructured(amount)
torch.nn.utils.prune.L1Unstructured(amount)
torch.nn.utils.prune.RandomStructured(amount, dim=-1)
torch.nn.utils.prune.LnStructured(amount, n, dim=-1)
torch.nn.utils.prune.CustomFromMask(mask)
torch.nn.utils.prune.random_unstructured(module, name, amount)
torch.nn.utils.prune.l1_unstructured(module, name, amount, importance_scores=None)
torch.nn.utils.prune.random_structured(module, name, amount, dim)
torch.nn.utils.prune.ln_structured(module, name, amount, n, dim, importance_scores=None)
torch.nn.utils.prune.global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs)
torch.nn.utils.prune.custom_from_mask(module, name, mask)
torch.nn.utils.prune.remove(module, name)
torch.nn.utils.prune.is_pruned(module)
torch.nn.utils.weight_norm(module, name='weight', dim=0)
torch.nn.utils.remove_weight_norm(module, name='weight')
torch.nn.utils.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None)
torch.nn.utils.remove_spectral_norm(module, name='weight')
torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)
torch.nn.Flatten(start_dim=1, end_dim=-1)
torch.nn.Unflatten(dim, unflattened_size)
torch.nn.modules.lazy.LazyModuleMixin(*args, **kwargs)


================================================================================
FILE: dump\deeprel-torch\api_args.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f1edb97ceb3a42eea461338b4f95d420","collectionName":"api_args","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\argVS.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c32510f5cbd94d7a90545f554affe7a8","collectionName":"argVS","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\signature.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"990cd8022ea34e91a6fe6c2fc944e236","collectionName":"signature","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\similarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d8da3c9ae82b4a52b7c245bf438626a4","collectionName":"similarity","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.abs.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0052bd9cdf284c8a858e202f8f440728","collectionName":"torch.abs","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.absolute.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9aa3af80b6d74ebaacb5e1e9eeda93c7","collectionName":"torch.absolute","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.acos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a044c5ea54f040c8971e8796a761062e","collectionName":"torch.acos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.acosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6579cbad31734feeb8110078c2c726f6","collectionName":"torch.acosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"59c1651212b24c5e9f47e2401392ca2a","collectionName":"torch.add","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1741eff7e2764597913f5232a7ee63ac","collectionName":"torch.addbmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addcdiv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2db4eb1961174657aad82dcc5d3e50d0","collectionName":"torch.addcdiv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addcmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9caf97dac8514336bbaae2101f22f3aa","collectionName":"torch.addcmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dcc945b1dccc43498374fd982b4b2854","collectionName":"torch.addmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addmv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"44c54a7479044d2e887f76d412e876ce","collectionName":"torch.addmv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.addr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7c9b4286c5b54bfca9da4354f9d78494","collectionName":"torch.addr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.all.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c99f43160177406d95464a1e260aa358","collectionName":"torch.all","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.allclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b809fade8505440eb16103c066cb219e","collectionName":"torch.allclose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.amax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c229c88db0fe4507b897cef7ddd33f23","collectionName":"torch.amax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.amin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"15fa7f5ebe74437bba1be4e048f87576","collectionName":"torch.amin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.aminmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0e54080e9e1e4f97bb6ca1a2f4af843e","collectionName":"torch.aminmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.angle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d627ddf5a93c4bd98270c71d38be46e3","collectionName":"torch.angle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.any.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"847e54832c664294b975cc8ac9e4dad3","collectionName":"torch.any","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arange.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2601d588431b4cda88b11cbbca31d3e1","collectionName":"torch.arange","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arccos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"59189a38b90144af8d19c0827edfbd21","collectionName":"torch.arccos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arccosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17742251f07146f4ab9a902ad898b72a","collectionName":"torch.arccosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arcsin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a3ec48f852da4fd0ac8137bae01cbe00","collectionName":"torch.arcsin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arcsinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1ed44801c7784a93bcbb7bbb2c1df918","collectionName":"torch.arcsinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arctan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23766dabfa574f428090c27ba8834e52","collectionName":"torch.arctan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.arctanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56b4dd83883042f5be1ee5a1e612d048","collectionName":"torch.arctanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.are_deterministic_algorithms_enabled.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"506519eb87e64e9e9e195d299631f4eb","collectionName":"torch.are_deterministic_algorithms_enabled","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.argmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"702d33a40cb5409b8c1313781b6548a4","collectionName":"torch.argmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.argmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8bec4c809776466cb2944c94089d8fb5","collectionName":"torch.argmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.argsort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6fe1fd11a3474f52947db1b1704621c1","collectionName":"torch.argsort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.asin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"29024b4c16474a5e8ee48564f709888c","collectionName":"torch.asin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.asinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3f395aa95ef746f29310b86d58975708","collectionName":"torch.asinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.as_strided.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bb94356a9f2f4af2938ff807365f4c98","collectionName":"torch.as_strided","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.as_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"481e006be75e401197e1848f7e91c9be","collectionName":"torch.as_tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5860ed7b74d1413cac813c0e8db808d9","collectionName":"torch.atan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atan2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a96cccfcd802480cb08c7a44a2e7e4a6","collectionName":"torch.atan2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9220f532b7104c10be7020ea783df17e","collectionName":"torch.atanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atleast_1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2f5df201bf5d412796aea09b8ff4090b","collectionName":"torch.atleast_1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atleast_2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bc967f8419284fa89349ebf848d0ea43","collectionName":"torch.atleast_2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.atleast_3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c61ea5dc21b744b69f2a180111dc8197","collectionName":"torch.atleast_3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.baddbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ce0cccac18f493eb1e7338a0ffd257e","collectionName":"torch.baddbmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bernoulli.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9bfb5679377642b782af80e7b8724f49","collectionName":"torch.bernoulli","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bincount.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"956218eda560480284a8fe2696a1fa27","collectionName":"torch.bincount","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"303f1497cd014558b94fe819a22c95c0","collectionName":"torch.bitwise_and","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_left_shift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6350fdf790ef4e74a4e02a24eb5b2fbf","collectionName":"torch.bitwise_left_shift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"51dc9a80bc894cb586552ed9e7d88889","collectionName":"torch.bitwise_not","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b6558fbf926414989f579a5273a3b35","collectionName":"torch.bitwise_or","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_right_shift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"51a34892a2534c188a15e2f533a43b1c","collectionName":"torch.bitwise_right_shift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bitwise_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"97f1ba7abcdf46ab9ac195ea76cc0ce8","collectionName":"torch.bitwise_xor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.block_diag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb4dcdb16c95485b8e6225a446fa3d5c","collectionName":"torch.block_diag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1dd74d23113841e684a72eb7589f0c20","collectionName":"torch.bmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.broadcast_shapes.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d7f3f5ebffc4ec88c50f397c0ec4ba8","collectionName":"torch.broadcast_shapes","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.broadcast_tensors.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2639bf0fe0264d6ea9782fb7c6e8cf65","collectionName":"torch.broadcast_tensors","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.broadcast_to.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"58b846fbe988436fbd84394aa0f33839","collectionName":"torch.broadcast_to","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.bucketize.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"91f512df6cce4110879fb4a190055da1","collectionName":"torch.bucketize","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.can_cast.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a4303b3d36d149c3977e94538bd97d41","collectionName":"torch.can_cast","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cartesian_prod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"38505abea9a7462cb35edb16ed650925","collectionName":"torch.cartesian_prod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95576046e832433080f1186020930acd","collectionName":"torch.cat","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cdist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b583aacffdd4b139f61dbe3d840d9ed","collectionName":"torch.cdist","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ceil.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"706ba5c722904294abd4576eabd5ac4f","collectionName":"torch.ceil","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.chain_matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4da06ed21e0a44f9b5244e0e36f1e12d","collectionName":"torch.chain_matmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cholesky.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"15ba82f790ef4bf987e9ef52ead9efa9","collectionName":"torch.cholesky","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cholesky_inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ab82214883d64223bd251088ca1a76dd","collectionName":"torch.cholesky_inverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cholesky_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"48f5874f30d245ff9a176e44fab6d19c","collectionName":"torch.cholesky_solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.chunk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"35cca6009dd44bf1b0c12ffec93178f2","collectionName":"torch.chunk","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.clamp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ea378f26967f421eb54db47f370c44b0","collectionName":"torch.clamp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.clip.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"98fe3100668f486fb6223006fb7a5681","collectionName":"torch.clip","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.clone.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc163de2ff8f4dc1920b7cac13b2d83d","collectionName":"torch.clone","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.column_stack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b9936381f7ef48ed902f00465da0d103","collectionName":"torch.column_stack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.combinations.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ed017a4062f404b938e7360a296678a","collectionName":"torch.combinations","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3ed51bf246034408bb305965f961ed16","collectionName":"torch.complex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"91a9da9738f94746914ba1d242b6de6c","collectionName":"torch.conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.conj_physical.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e7bb4531fc864db7898b4da32f94a33c","collectionName":"torch.conj_physical","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.copysign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"884d1c17ce464590b52f615ad2ed3643","collectionName":"torch.copysign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.corrcoef.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4f7d61905aa34eb899db56006bf936a7","collectionName":"torch.corrcoef","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"91ea3adddb8a40318255ffeaa49562d4","collectionName":"torch.cos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"236ebba61c874077a1247c949818a201","collectionName":"torch.cosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.count_nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"edeb8f793ee448db8e816c7f4607d724","collectionName":"torch.count_nonzero","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cov.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"076bcc04ca7b4e27be70ee42b1aa57c3","collectionName":"torch.cov","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cross.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23ea32c5568e48a0a4d17043a1eec8c4","collectionName":"torch.cross","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cummax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"833745e93fe64791a1a4b58aebf95115","collectionName":"torch.cummax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cummin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bfeff8f641474d86ade716934f3cbbc6","collectionName":"torch.cummin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cumprod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc1650bcd2ac41398fdc5ca1d57641ef","collectionName":"torch.cumprod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.cumsum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9145c430f3e64e76bc2388ff4b70a1fa","collectionName":"torch.cumsum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.deg2rad.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1cdac1928f854f7c9cc3db06eb0beb93","collectionName":"torch.deg2rad","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.det.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57088461e484492f8a009eef879777a5","collectionName":"torch.det","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.diag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b1ccf21d32b146eb8da69784548d10ff","collectionName":"torch.diag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.diagflat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5b0aa2868d294bb3947782084e8b2792","collectionName":"torch.diagflat","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.diagonal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0a276974d11d494e97dc59a9c5b6e4fc","collectionName":"torch.diagonal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.diag_embed.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ab95f5b59f134fdbacd47e949c485ece","collectionName":"torch.diag_embed","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.diff.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"680c3962fdfa4d6cb4a6683182f043b7","collectionName":"torch.diff","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.digamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9007b7547c74dc7958fefca34feef6a","collectionName":"torch.digamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.dist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a953e9abdc3458bb79c48c57960e118","collectionName":"torch.dist","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.AbsTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"daa8d8cbebf54ab5b370e0e5a1e035b1","collectionName":"torch.distributions.transforms.AbsTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.AffineTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8dfc955bdaef4f238437a969cc0d2229","collectionName":"torch.distributions.transforms.AffineTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.CorrCholeskyTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad29ad4badb14e358cf05bae30508ae0","collectionName":"torch.distributions.transforms.CorrCholeskyTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.ExpTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e63d077e32844b78374e5bc79e2f66d","collectionName":"torch.distributions.transforms.ExpTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.PowerTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1fa6c922bb3542c2b1be4a73a10e8af4","collectionName":"torch.distributions.transforms.PowerTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.SigmoidTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f1f94f58d53d4582a16055a2a0534490","collectionName":"torch.distributions.transforms.SigmoidTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.SoftmaxTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"735037553821458bae7b9180707f8cf0","collectionName":"torch.distributions.transforms.SoftmaxTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.StickBreakingTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b395103a221d4604b860255405a7e8ae","collectionName":"torch.distributions.transforms.StickBreakingTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.distributions.transforms.TanhTransform.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0958b83ad448476290968b1a47ea8463","collectionName":"torch.distributions.transforms.TanhTransform","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.div.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"be3d25b0f41842d3ae37ae0c516a6d83","collectionName":"torch.div","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3cc07da4984947d391462a587cdcb8e7","collectionName":"torch.divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.dot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9eab43d05ed54c499a77a0574cd2a9ee","collectionName":"torch.dot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.dsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2f6b1c97bc924e11a74e77db09748680","collectionName":"torch.dsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.dstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db94a32bddfe4a46a81f14193dc7f562","collectionName":"torch.dstack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.eig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"922fcec66f164fb7a31dc2581833161a","collectionName":"torch.eig","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.einsum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5cec564595a049d4917dd3585412228a","collectionName":"torch.einsum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.empty.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f5089e9f5643432e8d0b493bea425e08","collectionName":"torch.empty","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.empty_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"501591cebe9b4e7e9ac51f973511dcd1","collectionName":"torch.empty_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.empty_strided.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9cc424c4087c4d38896122ff08e333ad","collectionName":"torch.empty_strided","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.eq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7b6a418431ef454b9c7fb39c7551d210","collectionName":"torch.eq","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0265d5ea32d347a4811a6b61519909b8","collectionName":"torch.equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.erf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f6ed5c6ca39549e48c60243ed4892a3f","collectionName":"torch.erf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.erfc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"306c6dd584e94dbe8c59ceaf600e8a9d","collectionName":"torch.erfc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.erfinv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2db5903837d24f3ca65f2af15e2f2b3c","collectionName":"torch.erfinv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7c38489dfcee41908265182b51aead68","collectionName":"torch.exp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.exp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"35cdeb3c5ffa4bd9bc7e5bd91b965bd9","collectionName":"torch.exp2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.expm1.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d067101886ec48b99cbe2c531742eef1","collectionName":"torch.expm1","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.eye.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4858b7c1f52b443ea4fa618e74c836ea","collectionName":"torch.eye","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.fft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"42ba59fd171d48158f26833214552351","collectionName":"torch.fft.fft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.fft2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e679427b94364ceea415f1aff6c6fe40","collectionName":"torch.fft.fft2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.fftn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0e07bb288a774988add739a716f2bc9d","collectionName":"torch.fft.fftn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.fftshift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fbdb086249cf423581c2e97ccee76508","collectionName":"torch.fft.fftshift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.hfft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"15d33b4870b04870b8c8f7af1f1c7d8b","collectionName":"torch.fft.hfft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.ifft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"868a2e3ffa8c49ae8eb1f844b648b30a","collectionName":"torch.fft.ifft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.ifft2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"019c2cde0eab4df29d4e9fb14c32fed4","collectionName":"torch.fft.ifft2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.ifftn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"00bf1d58065b4546b3f3411d63c5302b","collectionName":"torch.fft.ifftn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.ifftshift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c6b521ac82db40399f4541a9b7b2f6af","collectionName":"torch.fft.ifftshift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.ihfft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d63caf6a8663411e8080efac4151a783","collectionName":"torch.fft.ihfft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.irfft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"04fd08312ef9476eb2947cd566471e7c","collectionName":"torch.fft.irfft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.irfft2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"737b8961b1d84eefaf7e53c0ad48cddb","collectionName":"torch.fft.irfft2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.irfftn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5b151e49e2b3438da7e9ab37f4aaef9b","collectionName":"torch.fft.irfftn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.rfft.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7bdeea064e5c41a588e2cbe4cea1b97b","collectionName":"torch.fft.rfft","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.rfft2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a8f362751064e9298d0e272391f2e0e","collectionName":"torch.fft.rfft2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fft.rfftn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c060ae3ba69240a38c5174a75254e5a4","collectionName":"torch.fft.rfftn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fix.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c3cb856840446bab0469bd7aab8760c","collectionName":"torch.fix","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.flatten.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5f6bd38347e4e55906a29e9ca685b3b","collectionName":"torch.flatten","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.flip.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a0f7cb7a7e845c895dd356175787066","collectionName":"torch.flip","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fliplr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"872671a4ecf347118fadd4c3453b8142","collectionName":"torch.fliplr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.flipud.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52404061162c4a8ebfaba8d8ba1c3c32","collectionName":"torch.flipud","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.float_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba6e10bb842d4688812f713bda21903a","collectionName":"torch.float_power","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.floor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b507a1a29b514b2b816748a7c0e922b9","collectionName":"torch.floor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.floor_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"53287fefe00c4678a989353f32038f90","collectionName":"torch.floor_divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb3584c1c8ee4a249271e70e9b70e3ba","collectionName":"torch.fmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c064025c5c23493d938218801d754cec","collectionName":"torch.fmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.fmod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8221df9071b046ff93b330b3125b509e","collectionName":"torch.fmod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.frac.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"46be330c089e413693f8b0518c2b5bfc","collectionName":"torch.frac","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.full.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7e7c1b4611874872957b18b141eed9b3","collectionName":"torch.full","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.full_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4d7e193aabb74fbc96e8322f38907650","collectionName":"torch.full_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.gather.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d836d08bd2b6403d8d4e003222dc04bf","collectionName":"torch.gather","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.gcd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4c5d98b2bc4f40c4822cda2c5096fdbe","collectionName":"torch.gcd","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ge.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"04e601591acc4c7081284307c564b11e","collectionName":"torch.ge","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ger.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"00e38e9491af45a696d833940b0a5ca8","collectionName":"torch.ger","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.get_rng_state.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8ab1934ea2bc4a76864c15fa9c7a4a4d","collectionName":"torch.get_rng_state","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.gradient.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4276c2c47ed54aa8818e6cbd3573ce30","collectionName":"torch.gradient","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.greater.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f67e89439a604b5a90ba5fd01470a006","collectionName":"torch.greater","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.greater_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bb16a39db32f4b2592069367423bd2e0","collectionName":"torch.greater_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.gt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9012402d513d4c46bf3e0e8c9a06ef57","collectionName":"torch.gt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.heaviside.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9d53cc77bc10416cb9b575db9fa6e42f","collectionName":"torch.heaviside","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.histc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9dfb818feebe44a18fe88c741003412d","collectionName":"torch.histc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.hsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1f15ee423dc5438c844ec9657114a45b","collectionName":"torch.hsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.hspmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b54ee30761d4ffbb8d91da1e30399ab","collectionName":"torch.hspmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.hstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23dcb6384ff14cf18b51502f8b87d5f5","collectionName":"torch.hstack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.hypot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95a042b625204449af6f8e9ff56015f7","collectionName":"torch.hypot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.i0.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0e4d51d6da0c40ac8974701133335e2d","collectionName":"torch.i0","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.igamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a63dd7d99f80440784fa368ddf6cc1cc","collectionName":"torch.igamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.igammac.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95570a1d2a4841549e536e75d2edff46","collectionName":"torch.igammac","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.imag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a065f0b5205a42a9b9b8af70bb980d7d","collectionName":"torch.imag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.index_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b94f75b2ca724ad18bd859ba2591d975","collectionName":"torch.index_select","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.inner.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3123323212fd489aac341f7b6a6d90f5","collectionName":"torch.inner","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5dc967f2a4db40d4b4393677a80c7b82","collectionName":"torch.inverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"72af1ffd78b044cdb12dcaabb658ba99","collectionName":"torch.isclose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isfinite.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"176f0f0492fa41e092741f5fc81736ce","collectionName":"torch.isfinite","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ecb42899076246c3936de1d3130c193f","collectionName":"torch.isinf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isnan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f3f501dc2f9843778dfbdf313d436d2f","collectionName":"torch.isnan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isneginf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7c8cf726e69640948b2879183ffe839a","collectionName":"torch.isneginf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isposinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c8e42998e2345daa41d46fc9a94823d","collectionName":"torch.isposinf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.isreal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"da44ac9ef2304b90a2a47e924c50e784","collectionName":"torch.isreal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3f336467e6864f3aa3655481dbd8566f","collectionName":"torch.is_complex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f18c8388dd71475499d38a27e5b5d0b6","collectionName":"torch.is_conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_floating_point.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3f26e0d86a74436da76dd94fb09046ba","collectionName":"torch.is_floating_point","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"932f36708a4b4d6e9b71674dc67f1d78","collectionName":"torch.is_nonzero","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_storage.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"09ff96b6edfe449a83ee08bad04f0a44","collectionName":"torch.is_storage","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.is_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"30af123340ad43b2a6d53a73112900af","collectionName":"torch.is_tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.jit.script_if_tracing.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c3814217691a448b90f94153fda13c9f","collectionName":"torch.jit.script_if_tracing","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.kron.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e203162014d6467cb4509ee39a414d34","collectionName":"torch.kron","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.kthvalue.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"69965eb9c7f74d18bb90b90bb8c901ae","collectionName":"torch.kthvalue","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lcm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c760c742c004729b8cc58397a873ed0","collectionName":"torch.lcm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ldexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3eebe7fe26034506b440a02b753959c4","collectionName":"torch.ldexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.le.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"804f3af533db4e41bf4a6ac366a76ba2","collectionName":"torch.le","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lerp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db453d097d3e4549ac863b3f90624a08","collectionName":"torch.lerp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.less.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fa03f875e04141839603f1ab6165932b","collectionName":"torch.less","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.less_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c3d7adffe1b441bc8108a3bd2f1e41a8","collectionName":"torch.less_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"46da2339cc414e34a4640a1d41f3b97f","collectionName":"torch.lgamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.cholesky.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c4b30d31a9324622bdfad42583dda0c1","collectionName":"torch.linalg.cholesky","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.cholesky_ex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"77dcd77a325d4521a52d2c8cf50453e0","collectionName":"torch.linalg.cholesky_ex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.cond.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2cb967782bf248b6b07e0a83def7f4ed","collectionName":"torch.linalg.cond","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.det.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"671b6b04a06248249dd0141e176e9f5c","collectionName":"torch.linalg.det","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.eig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5e11d29345a74334945110a182d6212b","collectionName":"torch.linalg.eig","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.eigh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"010473ff3d53498997661ccef0bb2337","collectionName":"torch.linalg.eigh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.eigvals.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dc8edfa02e9f4543ade45d6e0f5cfe6c","collectionName":"torch.linalg.eigvals","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.eigvalsh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b93299f701084b279af48c1be5878c69","collectionName":"torch.linalg.eigvalsh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.inv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"16eb5b6f07f548b38822a34fd098d4ed","collectionName":"torch.linalg.inv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.inv_ex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17dc17ca66344d3ebeffe44711b54929","collectionName":"torch.linalg.inv_ex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.lstsq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"81b2ef3f496c4f2d8586b09f4f733155","collectionName":"torch.linalg.lstsq","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"322e88f0c7a74ca593bac30bcfde3139","collectionName":"torch.linalg.matmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.matrix_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"425362ca2a3f4eab91d72b207d8b920f","collectionName":"torch.linalg.matrix_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.matrix_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a2caa068df04fd880d93fcad3b60924","collectionName":"torch.linalg.matrix_power","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.matrix_rank.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d14fbc4bd6544db39e3a55b10b5a33c3","collectionName":"torch.linalg.matrix_rank","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9f1150c31fe74f27abea313e766f637e","collectionName":"torch.linalg.norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.pinv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"51feb3973dd74e55bb404dcb4d4f807e","collectionName":"torch.linalg.pinv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.qr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ba598956c074a86aa1d327abe5e5701","collectionName":"torch.linalg.qr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.slogdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bda8828ed1ab4ef687d93f328126b504","collectionName":"torch.linalg.slogdet","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4110d4dff2384cefa5ecebfe7fefdcfe","collectionName":"torch.linalg.solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.svd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"76d24cc160fc48e7986568d29abb9141","collectionName":"torch.linalg.svd","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.svdvals.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"32d15a57b5c14a01a2c3329643b4f3d4","collectionName":"torch.linalg.svdvals","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linalg.vector_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1232a9cab3ba4a188c29c753c1808f40","collectionName":"torch.linalg.vector_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.linspace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b0f3d0a493754cfe9bc9e3041815166c","collectionName":"torch.linspace","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lobpcg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6b376dc23e694a008ab751bbc0165f76","collectionName":"torch.lobpcg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.log.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9178e1f2078e42d99fe022a91dc17713","collectionName":"torch.log","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.log10.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc6f3af91baa439fba88e7f023302df8","collectionName":"torch.log10","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.log1p.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"833ceda46de04ede927400907a2d9c9c","collectionName":"torch.log1p","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.log2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a8776c9982b3484f8eab30222b030d25","collectionName":"torch.log2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logaddexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7cd8dccae2724dc8ae77b203ce3e026b","collectionName":"torch.logaddexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logaddexp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b2b01b54b5824b3489950a22e1a19e94","collectionName":"torch.logaddexp2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logcumsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"05a4dfbef3bf4874a167aa4529f843e3","collectionName":"torch.logcumsumexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"733d32356b604899a42e144f87912d88","collectionName":"torch.logdet","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logical_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4d93bfb9ea974e978d2c57bfc343a23e","collectionName":"torch.logical_and","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logical_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9539d3c29da242f68553fde88f4c491e","collectionName":"torch.logical_not","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logical_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0f63a8c77467405eb0df99c0988d6c10","collectionName":"torch.logical_or","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logical_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6de1bc28d01e4d7bb417fce4630133f7","collectionName":"torch.logical_xor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"81bdd5fd925549a59e35d6e6971479ec","collectionName":"torch.logit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logspace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a663adb45724427b972b0c3599200a3","collectionName":"torch.logspace","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.logsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f05a3df0d7a649a6803f33a1815a5105","collectionName":"torch.logsumexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lstsq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e2e1bef2c824f62b089a1e646440862","collectionName":"torch.lstsq","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5578bd169c8f4123a4917da5e3c88710","collectionName":"torch.lt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d59022c111b745f89ef2bc0c602faed9","collectionName":"torch.lu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.lu_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"01b4f31f6c0141bca1733464634d2235","collectionName":"torch.lu_solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.masked_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dd3499f16f2f47efa31e1ccc0cf49daa","collectionName":"torch.masked_select","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e440a7fab5046fdae8f34a88c282baa","collectionName":"torch.matmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.matrix_exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ea1442e73ff40568fc06c37c886e9ca","collectionName":"torch.matrix_exp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.matrix_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c08a53a1091a4fdb81982fa6e68add5c","collectionName":"torch.matrix_power","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.matrix_rank.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"98cfc4a37f00426198f10ba5ac1610f8","collectionName":"torch.matrix_rank","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.max.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"594208d5ad434568b4098171e44618dc","collectionName":"torch.max","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.maximum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"691332f698f84945b41282ceaa4295af","collectionName":"torch.maximum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a69a1a15a0ea492582df1c7f327a2b49","collectionName":"torch.mean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.median.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"15a2e816caf843488e053d652487846f","collectionName":"torch.median","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.meshgrid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"73bc06af89a04d7f8ef1bccbc8c4b3b5","collectionName":"torch.meshgrid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.min.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8e0c371b04fc4dabbf1d7e1ea2226251","collectionName":"torch.min","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.minimum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e2715b2e00e4452db58f0f758fb5c793","collectionName":"torch.minimum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2bb4e3394f05444daec5e52f9d776bd9","collectionName":"torch.mm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mode.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17b77aacf6e64ac287e445a2004cb6ab","collectionName":"torch.mode","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.moveaxis.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ad5a5df3a17433bbd8626ae6195114c","collectionName":"torch.moveaxis","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.movedim.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e70f60c367804816b80b833c6b1d3970","collectionName":"torch.movedim","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.msort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4997a6babf5c4b4a867b570942683ece","collectionName":"torch.msort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad36fef65920476ebbdbbddf511e87f8","collectionName":"torch.mul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.multinomial.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2799e567fbe341bdb730b4567e0fef43","collectionName":"torch.multinomial","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.multiply.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"01bed96f43c74aee86d5ae023c907179","collectionName":"torch.multiply","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9d1fa594cca44619a1c1dd29715dbda7","collectionName":"torch.mv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.mvlgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a89694739fd046fab3d850c7858a723d","collectionName":"torch.mvlgamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nanmean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b85a20b1bff84e6f8d96ef394186bdf9","collectionName":"torch.nanmean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nanmedian.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c6f52cfc54845218191e53f6a4a6480","collectionName":"torch.nanmedian","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nanquantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"479c283320f24c488ce38df0f22d731f","collectionName":"torch.nanquantile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nansum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52ca61296833491cae76b8c45aefe54a","collectionName":"torch.nansum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nan_to_num.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de8e27621eec42ef97e0b3c0cdd40e57","collectionName":"torch.nan_to_num","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.narrow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"380e7e854fc247b19ac1db8d2d730c03","collectionName":"torch.narrow","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ne.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6426b61823d24f9f98e03ded925fd2a1","collectionName":"torch.ne","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.neg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bb0f4b5e935f47dfbdf90ee83f88390b","collectionName":"torch.neg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.negative.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2be57c39ddd14a3889385c9aae4a67c5","collectionName":"torch.negative","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nextafter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0c559c45065843a3afe0a79190c8596f","collectionName":"torch.nextafter","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveAvgPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f45620ef7add47d995db55281200c053","collectionName":"torch.nn.AdaptiveAvgPool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveAvgPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c0a9abdbd7304ea0aa426d1a16ba2270","collectionName":"torch.nn.AdaptiveAvgPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveAvgPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c5cc92a8f4ce4697a5396656a9575397","collectionName":"torch.nn.AdaptiveAvgPool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveMaxPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"06e93ae2b72e4a659b1b44f8a3d9e721","collectionName":"torch.nn.AdaptiveMaxPool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveMaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bccb97adc78a4809a85b229673603213","collectionName":"torch.nn.AdaptiveMaxPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AdaptiveMaxPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ea4a16b91eb44f14a9d5484a3a19c687","collectionName":"torch.nn.AdaptiveMaxPool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AlphaDropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5e122af9ca0b4704926a282cb263bd19","collectionName":"torch.nn.AlphaDropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AvgPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"866e84aa75874d12b605d141c1f22690","collectionName":"torch.nn.AvgPool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AvgPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0a96328668de4733b9eff0e81845253c","collectionName":"torch.nn.AvgPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.AvgPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95e2e215d27b4d89b76bfd65e72043b3","collectionName":"torch.nn.AvgPool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.BatchNorm1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b4a013841394d0c92fd40df85c9ef53","collectionName":"torch.nn.BatchNorm1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.BatchNorm2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e0dc79fef5e54b35b53671d48b8ebd1a","collectionName":"torch.nn.BatchNorm2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.BatchNorm3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"08c1592f4bcd4845aa9bbb8c8d5249f7","collectionName":"torch.nn.BatchNorm3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.BCELoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"957f0622a144417491a21da2dd319a7f","collectionName":"torch.nn.BCELoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.BCEWithLogitsLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"28019a0fcfae4738a11781086cfbb111","collectionName":"torch.nn.BCEWithLogitsLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Bilinear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b1e4cbbbcfd8423e8bf1a43d195401b6","collectionName":"torch.nn.Bilinear","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.CELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"841579d22e0a43c590d6c20a9bd3604e","collectionName":"torch.nn.CELU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ChannelShuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"12de8cc980fd4e03922fdb606860f505","collectionName":"torch.nn.ChannelShuffle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConstantPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e8bd111687d04391b9876914a53f35d8","collectionName":"torch.nn.ConstantPad1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConstantPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"94eac0138e9546ccac2b76737b975e31","collectionName":"torch.nn.ConstantPad2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConstantPad3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de7f24d087a7442299a6328de64f34db","collectionName":"torch.nn.ConstantPad3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Conv1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c3f269ae43554764ae26520081faadc3","collectionName":"torch.nn.Conv1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Conv2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"02cd3651b4924ebc94e237d83b439f30","collectionName":"torch.nn.Conv2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Conv3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fa3e57774db346bcac51adba7164277b","collectionName":"torch.nn.Conv3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConvTranspose1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bfed4470d93e4c0299aacda81011082d","collectionName":"torch.nn.ConvTranspose1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConvTranspose2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33bde011ff354788bb11c760cc5f9ac5","collectionName":"torch.nn.ConvTranspose2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ConvTranspose3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f0048b5d8f534b0eafe4362d970b59b9","collectionName":"torch.nn.ConvTranspose3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.CosineSimilarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cf0af53b7b164fbf91944b6f301902d6","collectionName":"torch.nn.CosineSimilarity","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.CrossEntropyLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dfca09bb18e949c9a2e97528533cba81","collectionName":"torch.nn.CrossEntropyLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.CTCLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36a1ce4698f540f7a0dad54d271b89e2","collectionName":"torch.nn.CTCLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"387e720f555847c2a9933acf2ef2abbe","collectionName":"torch.nn.Dropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Dropout2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2fd57b3a24c24cd2b45a2244956a1d48","collectionName":"torch.nn.Dropout2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Dropout3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2f89a9c633b743998a74190ad8238871","collectionName":"torch.nn.Dropout3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3867f5b1a8be4d84962cdcc8b0b1539d","collectionName":"torch.nn.ELU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Embedding.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c7eb0204dcd8406e87f54fae75d77507","collectionName":"torch.nn.Embedding","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.EmbeddingBag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f669272bf59340ceb9cf775e7c1aeecd","collectionName":"torch.nn.EmbeddingBag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.FeatureAlphaDropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"84cbaeef9e8b4781ba8923d8ee34e854","collectionName":"torch.nn.FeatureAlphaDropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Flatten.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b3d1d87de9cd47deaee90ab6464f9778","collectionName":"torch.nn.Flatten","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Fold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"35f24be2f9a64d7bacb1dc67c91f3cd1","collectionName":"torch.nn.Fold","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.FractionalMaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"16d028c448bf43b9a3684fd54481f243","collectionName":"torch.nn.FractionalMaxPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_avg_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fd5e843280864c6ba4f68fa570168164","collectionName":"torch.nn.functional.adaptive_avg_pool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_avg_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1dbee18b4feb4be39dc3847a0acd46ef","collectionName":"torch.nn.functional.adaptive_avg_pool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_avg_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"79d3295ddfed404297493951b21364a0","collectionName":"torch.nn.functional.adaptive_avg_pool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_max_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"143ec58cd4904d029f8d5634a47d0864","collectionName":"torch.nn.functional.adaptive_max_pool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_max_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d041cb82936344468c5dfc2cdbe7128a","collectionName":"torch.nn.functional.adaptive_max_pool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.adaptive_max_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d139d75ba0c24f35b507d10a5b17b9a8","collectionName":"torch.nn.functional.adaptive_max_pool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.alpha_dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"444d66550c9a4035ac003baa9859cd6e","collectionName":"torch.nn.functional.alpha_dropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.avg_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb60c431e4b34f319d7e3abbb7d8f597","collectionName":"torch.nn.functional.avg_pool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.avg_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"934ba3cf26714f118d6851b5760e26f6","collectionName":"torch.nn.functional.avg_pool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.avg_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36d41109842f4ac4a2e8fa98012b1459","collectionName":"torch.nn.functional.avg_pool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.batch_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db899514b67d47b8b12363ce722ec327","collectionName":"torch.nn.functional.batch_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.bilinear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56686f1221de498c8221cd22958fcc6f","collectionName":"torch.nn.functional.bilinear","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.binary_cross_entropy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3399fb62e79847d7872967ca286d504b","collectionName":"torch.nn.functional.binary_cross_entropy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.binary_cross_entropy_with_logits.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db9a3204b7bc43a3b4ad1925e3e4e1a4","collectionName":"torch.nn.functional.binary_cross_entropy_with_logits","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.celu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3bac283f604141a5a6c0c4f4242f1e18","collectionName":"torch.nn.functional.celu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e270cca02abe4bc0b9840e1e42860c2d","collectionName":"torch.nn.functional.conv1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5ebe98ca713d474ea2491311e837fbcb","collectionName":"torch.nn.functional.conv2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"30a210f6e6124de5a37153c4e99175af","collectionName":"torch.nn.functional.conv3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv_transpose1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"358a1a46fffa48c9b1a0dca074802b23","collectionName":"torch.nn.functional.conv_transpose1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv_transpose2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7f3b75cbeff54a31ab5b467109f74ca1","collectionName":"torch.nn.functional.conv_transpose2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.conv_transpose3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9212d0b5eb784296af462ee4a67ab1dc","collectionName":"torch.nn.functional.conv_transpose3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.cosine_similarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5a2e5634f2f4ef2bcbf54b9ecb0bbe4","collectionName":"torch.nn.functional.cosine_similarity","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.cross_entropy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac1e8e1ca9124c7ea56d0c7f569e3a30","collectionName":"torch.nn.functional.cross_entropy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.ctc_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"64b339975a17467e95d8e1e6ce927d69","collectionName":"torch.nn.functional.ctc_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"863fbaba5fae4d25b21095c5ccfc9143","collectionName":"torch.nn.functional.dropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.dropout2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fd115219f43a40bd8f07236a98f55258","collectionName":"torch.nn.functional.dropout2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.dropout3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ce97f9fc80954935a35c26796b3bdf1a","collectionName":"torch.nn.functional.dropout3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.elu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c8b06629872d4304a3d954386b53a7a8","collectionName":"torch.nn.functional.elu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.elu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"07e98391a4254193a70fa62f4e9fbde6","collectionName":"torch.nn.functional.elu_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.embedding.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bd809bf173d74a63b699f2e4aa77f483","collectionName":"torch.nn.functional.embedding","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.embedding_bag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a2b758abe2ac42b5b8abfd027e28655b","collectionName":"torch.nn.functional.embedding_bag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.feature_alpha_dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9823e0526d7f4190a1fa1ee5d9e6b3de","collectionName":"torch.nn.functional.feature_alpha_dropout","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.fold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db58dba2183949f7beb56663b58d5fa3","collectionName":"torch.nn.functional.fold","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.gelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"09d5a03a3538411b88727c3affc8801f","collectionName":"torch.nn.functional.gelu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.gumbel_softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"040ffe65f6114d34ab3dfa9444f0d9c6","collectionName":"torch.nn.functional.gumbel_softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hardshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"68f095845fe94897a65d77bad458f567","collectionName":"torch.nn.functional.hardshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hardsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e3f7eff7a05465abe24b28bbcfbf53e","collectionName":"torch.nn.functional.hardsigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hardswish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"13c20fb764144de481d64269eaeb3bb2","collectionName":"torch.nn.functional.hardswish","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hardtanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ec7c51bb7d0a4b8b8fb29b12aad2a798","collectionName":"torch.nn.functional.hardtanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hardtanh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a0e5effd4c784bb3820e3fdeb49de9e0","collectionName":"torch.nn.functional.hardtanh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.hinge_embedding_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d8b9aa1a51b44b0a61619dd78fbd507","collectionName":"torch.nn.functional.hinge_embedding_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.huber_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9a52d071c2ff4058bffd098807c2b6ea","collectionName":"torch.nn.functional.huber_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.instance_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f538eb44311b4e738dd4e96a8d007857","collectionName":"torch.nn.functional.instance_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.interpolate.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e797c743148a4d3fba7044f97ca3c97a","collectionName":"torch.nn.functional.interpolate","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.kl_div.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0168ed2cb17c40fe8d812c2133baa036","collectionName":"torch.nn.functional.kl_div","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.l1_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6412f37eada94de9b16436f21a49aba1","collectionName":"torch.nn.functional.l1_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.layer_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f181628ea16648d78b2d7dcd059009cb","collectionName":"torch.nn.functional.layer_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.leaky_relu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2915416f943645bcbb612bff73f0d0a1","collectionName":"torch.nn.functional.leaky_relu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.leaky_relu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7f8667dcdd544bd889f6dc75216f81b5","collectionName":"torch.nn.functional.leaky_relu_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.linear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dbc8ee0ca76048e58c620a27ff21443f","collectionName":"torch.nn.functional.linear","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.local_response_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4c009c9dcd9a4e1181c6b0e71b1d58f8","collectionName":"torch.nn.functional.local_response_norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.logsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"aa61217f0a0443ffbf151cb80e9eaa8d","collectionName":"torch.nn.functional.logsigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.log_softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5e9d032f5a9649a0aaf5210bfe129f7f","collectionName":"torch.nn.functional.log_softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.lp_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ef19c95f3a5449ed8021c1d10c216218","collectionName":"torch.nn.functional.lp_pool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.lp_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"00d2bba9e9f949378ef43ca8d697605c","collectionName":"torch.nn.functional.lp_pool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.margin_ranking_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"143255125ba04cd8aa781de7c3bfa081","collectionName":"torch.nn.functional.margin_ranking_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"76d7ea663d574b2486890bf1337723fe","collectionName":"torch.nn.functional.max_pool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7c5f26e286e34879be6bdcbdddbcbcff","collectionName":"torch.nn.functional.max_pool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"924b57f1f6084a0e8eb476c49b27ab78","collectionName":"torch.nn.functional.max_pool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_unpool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"89f2adc9c2d1498995fe2aad1fd32563","collectionName":"torch.nn.functional.max_unpool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_unpool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"545cfb3c06384825bf19f41d300821e0","collectionName":"torch.nn.functional.max_unpool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.max_unpool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"aaacde0ae429433bb3842729fb90b995","collectionName":"torch.nn.functional.max_unpool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.mish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"802acac16f254271bef77cd62a07afd6","collectionName":"torch.nn.functional.mish","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.mse_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9c0fb842720944629101f9afd6920e66","collectionName":"torch.nn.functional.mse_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.multilabel_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b4b990f534294f5cb95e51cce7b2b056","collectionName":"torch.nn.functional.multilabel_margin_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.multilabel_soft_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e2b1b951a9a74fb8894f718fa35d56b8","collectionName":"torch.nn.functional.multilabel_soft_margin_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.nll_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"533b7ba2ca09414884f2e154441b568f","collectionName":"torch.nn.functional.nll_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.normalize.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de6296831ae440c0a1c4160ffa2b3cd7","collectionName":"torch.nn.functional.normalize","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.pad.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d13c6b3e412d4dfd9f0b12fda1be8a84","collectionName":"torch.nn.functional.pad","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.pairwise_distance.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6f4844dd09ad44469aef1bc548eb73e4","collectionName":"torch.nn.functional.pairwise_distance","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.pixel_shuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"80841f710af44d21a4a7a1cdefda7345","collectionName":"torch.nn.functional.pixel_shuffle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.pixel_unshuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5de6b12b7b364d4583fe1670c4c8b3a5","collectionName":"torch.nn.functional.pixel_unshuffle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.poisson_nll_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5dce2ee4a59f4b6d878ed7e67f2dbdd8","collectionName":"torch.nn.functional.poisson_nll_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.prelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f34b7ab65ec40c89a6b89ca2c0b70e8","collectionName":"torch.nn.functional.prelu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.relu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ca723d71c5cc4a4795c79f53e9299bc5","collectionName":"torch.nn.functional.relu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.relu6.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cf2d44ebf4164e9d8cc4cf02eec1e8e3","collectionName":"torch.nn.functional.relu6","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.relu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95985402c95d49d382025458c03ddfa6","collectionName":"torch.nn.functional.relu_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.rrelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cad20eb4937d4f39815a477443dabcdf","collectionName":"torch.nn.functional.rrelu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.rrelu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5fb4623737f64088b4f62a4d9067534b","collectionName":"torch.nn.functional.rrelu_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.selu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96f186674a5740d6a99359ca4edb87da","collectionName":"torch.nn.functional.selu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e6ba716a1c7443e9ffadda1a0d1ae80","collectionName":"torch.nn.functional.sigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.silu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9e2c07674e3c40c0991283e7f7421454","collectionName":"torch.nn.functional.silu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.smooth_l1_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"20c73f1173034dc4b2ac7c3527917d78","collectionName":"torch.nn.functional.smooth_l1_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"371eb85dfb8c4c47a4afc64bc586369b","collectionName":"torch.nn.functional.softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.softmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"858cefd15f5a4a34b28458835b3544bb","collectionName":"torch.nn.functional.softmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.softplus.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"27f6312fc55644068336bfc1c1531d02","collectionName":"torch.nn.functional.softplus","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.softshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b5595ad55a724e2f813bb0a8d37e65bd","collectionName":"torch.nn.functional.softshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.softsign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cd71b3e2816e4cdb85999466248da58c","collectionName":"torch.nn.functional.softsign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.soft_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bae7d247bb53463bb6ec87adc51e2e25","collectionName":"torch.nn.functional.soft_margin_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"18b5b67eee0f4f78b2a809499e6d2683","collectionName":"torch.nn.functional.tanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.tanhshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2933da3ddd93409885c5eb0222a1ce27","collectionName":"torch.nn.functional.tanhshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.threshold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3424d83584c34bf3944be65166c2e9dc","collectionName":"torch.nn.functional.threshold","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.threshold_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f452f509edb044c5beda655b18a78876","collectionName":"torch.nn.functional.threshold_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.triplet_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9fc6b562cd3240948e4e7ef4d27e7762","collectionName":"torch.nn.functional.triplet_margin_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.triplet_margin_with_distance_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a0be53261d7f44928f066d6bd1e80041","collectionName":"torch.nn.functional.triplet_margin_with_distance_loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.upsample.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b68ad1cf4a444ed7b0f146de9bfee752","collectionName":"torch.nn.functional.upsample","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.upsample_bilinear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bef0166e3be44f6091763c0cf1492dca","collectionName":"torch.nn.functional.upsample_bilinear","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.functional.upsample_nearest.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba277a0f41c54e3d9c52b1e1c3f6b3aa","collectionName":"torch.nn.functional.upsample_nearest","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.GELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d30a48cfef4945fab7566106093d8ad9","collectionName":"torch.nn.GELU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.GLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"357201ad64074e2e9a3a5ac19d6f6ee0","collectionName":"torch.nn.GLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.GroupNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e85a199267264ffea9c7704fb06d8b4f","collectionName":"torch.nn.GroupNorm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.GRU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"141395e73e7b4b8d91cbb96bf27827e1","collectionName":"torch.nn.GRU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.GRUCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d75e53394afd4552a3b4fbd204e5f52a","collectionName":"torch.nn.GRUCell","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Hardshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8cfa00c718b0440fb63063d57740c1d2","collectionName":"torch.nn.Hardshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Hardsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"78b5033a4264418a8ad2eb2d28caf8fa","collectionName":"torch.nn.Hardsigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Hardswish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b291978fe06f4c4082e809e1209ceb5c","collectionName":"torch.nn.Hardswish","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Hardtanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4638cb8f2931400a91c33fa40aa4a262","collectionName":"torch.nn.Hardtanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.HingeEmbeddingLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"77ab7327d95c4ec08cbec6112b30aaca","collectionName":"torch.nn.HingeEmbeddingLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.HuberLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a76c75708134385b8c901e8780d9074","collectionName":"torch.nn.HuberLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Identity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"75af88871aed4d6fa8d73d6836104e7e","collectionName":"torch.nn.Identity","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.constant_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8e584ef921074f3395e774f91ba1855b","collectionName":"torch.nn.init.constant_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.eye_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9950f48767a4d0384a813ba13725545","collectionName":"torch.nn.init.eye_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.kaiming_normal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f02f479c56f74453850e411f39aec09d","collectionName":"torch.nn.init.kaiming_normal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.kaiming_uniform_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fa1ae6925f384a7bbdc556ea16709bb7","collectionName":"torch.nn.init.kaiming_uniform_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.normal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c5ed50025bd4402198a3495c528b3c3b","collectionName":"torch.nn.init.normal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.ones_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e69884c1315441f89b8f5057d576cd8f","collectionName":"torch.nn.init.ones_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.orthogonal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5b4772cbb6cc4cfc82abdcbbc99cc3fe","collectionName":"torch.nn.init.orthogonal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.sparse_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a1648b0b58a64a2abcd88592d968d985","collectionName":"torch.nn.init.sparse_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.uniform_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c903564e43124d7e9dd1ab049a2005c7","collectionName":"torch.nn.init.uniform_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.xavier_normal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2622b60d4711475a8635399a7e3d323c","collectionName":"torch.nn.init.xavier_normal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.xavier_uniform_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"df64a9dc47bc499781e05c53086b446f","collectionName":"torch.nn.init.xavier_uniform_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.init.zeros_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52d63b987ee34e6aac6a27316d9d8bb9","collectionName":"torch.nn.init.zeros_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.InstanceNorm1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9cf2e5283f544a27b68a03cd7bad467d","collectionName":"torch.nn.InstanceNorm1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.InstanceNorm2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"54407343e3fc47a7a9b58d1878d314e9","collectionName":"torch.nn.InstanceNorm2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.InstanceNorm3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57b40551cc6d4aa7bd75b16b41a32abc","collectionName":"torch.nn.InstanceNorm3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.KLDivLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c7c55a0739394cee9ca37bfe226acc71","collectionName":"torch.nn.KLDivLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.L1Loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"da861f5b51694fb2aa68c1afaa933bbb","collectionName":"torch.nn.L1Loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LayerNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"41409563f8224576991afaefb418a673","collectionName":"torch.nn.LayerNorm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LeakyReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9fcd30454eff434ca9faa67aa75213a5","collectionName":"torch.nn.LeakyReLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Linear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"edf55c01a0ab4660bedff6cc903b78f6","collectionName":"torch.nn.Linear","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LocalResponseNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"54001331d2d84affa13c45924c7d2a1a","collectionName":"torch.nn.LocalResponseNorm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LogSigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9d82fd506a643588011799eafb43e70","collectionName":"torch.nn.LogSigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LogSoftmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d1ad2f4133314636bbbae2d8659a0093","collectionName":"torch.nn.LogSoftmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LPPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4288b680f8044066bf4c62dfd6aa6896","collectionName":"torch.nn.LPPool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LPPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1c924373f3a74f96887a4b2379d6633c","collectionName":"torch.nn.LPPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LSTM.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c718490183104093bc5a014dd1ad8c50","collectionName":"torch.nn.LSTM","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.LSTMCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"411d45cf4fc140f3a510abd6c553c70b","collectionName":"torch.nn.LSTMCell","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MarginRankingLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"98b6c248bfec49fca0139f49f63e690f","collectionName":"torch.nn.MarginRankingLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6911e59075c6478bb3022492d56b2788","collectionName":"torch.nn.MaxPool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7f7172320482483283483cc3e98bd427","collectionName":"torch.nn.MaxPool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"25200808bda1434bbcdd42d41cdf6d70","collectionName":"torch.nn.MaxPool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxUnpool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b3a96cbc64c4226beb28a5e46dd837a","collectionName":"torch.nn.MaxUnpool1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxUnpool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"41e602a5c42849498e319eb26d16addb","collectionName":"torch.nn.MaxUnpool2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MaxUnpool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e55b26ae10584da694ce43695a836254","collectionName":"torch.nn.MaxUnpool3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Mish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c7fbd5b9aa14474bceedd13226fd349","collectionName":"torch.nn.Mish","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MSELoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b0dee161986449b88f98bc725ad7b31","collectionName":"torch.nn.MSELoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MultiheadAttention.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4648ae91df824fea980e5fd3ab4fc07c","collectionName":"torch.nn.MultiheadAttention","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MultiLabelMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"714732918c6341a1b30b64e2ca1c03b3","collectionName":"torch.nn.MultiLabelMarginLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MultiLabelSoftMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"138b2f89d9f04f9782c9f2f3cbb2c301","collectionName":"torch.nn.MultiLabelSoftMarginLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.MultiMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1727e8724a704d1887562f6b3fd9f3d9","collectionName":"torch.nn.MultiMarginLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.NLLLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e42fbf0afffd4aac98575011a947d7e6","collectionName":"torch.nn.NLLLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.PairwiseDistance.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8bb5059cac224f958f4fdefee5588e47","collectionName":"torch.nn.PairwiseDistance","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.PixelShuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4ed71ef959124d82901949d16c41857c","collectionName":"torch.nn.PixelShuffle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.PoissonNLLLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f72820303dc145d6a7085c75af765a3d","collectionName":"torch.nn.PoissonNLLLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.PReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1d4ffa262f3849968b02ba8440d16690","collectionName":"torch.nn.PReLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReflectionPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f6cece0c71374597b3869d2a62c60d49","collectionName":"torch.nn.ReflectionPad1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReflectionPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d8ab723dc5784d3890e8f5c6016a5c53","collectionName":"torch.nn.ReflectionPad2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReflectionPad3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e76e927b88784e078dbedc66b383a1b7","collectionName":"torch.nn.ReflectionPad3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5cce60eb71b47d18a677c153f5ea386","collectionName":"torch.nn.ReLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReLU6.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"983d64d7ba5c4f6fbe48c2d601ce867d","collectionName":"torch.nn.ReLU6","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReplicationPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96dc16b86363478da503597717168e88","collectionName":"torch.nn.ReplicationPad1d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReplicationPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ecc755cc77e4699ba77c789e9514a22","collectionName":"torch.nn.ReplicationPad2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ReplicationPad3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0977a6c51979434c828ffffb57689b35","collectionName":"torch.nn.ReplicationPad3d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.RNN.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c3895f8caafb49e7af3d3a9f599cc048","collectionName":"torch.nn.RNN","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.RNNBase.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3df6b4db9c6d4ef1b9a6d3b2df3dbb2d","collectionName":"torch.nn.RNNBase","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.RNNCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"509787cdcd5142df8de61d0533634e21","collectionName":"torch.nn.RNNCell","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.RReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"75dc9c7015a140af899db325f763871c","collectionName":"torch.nn.RReLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.SELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fee79dcaf7af44f3b5f36d5b0260a952","collectionName":"torch.nn.SELU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"593f4fefa63f47b68e615c91a690745e","collectionName":"torch.nn.Sigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.SiLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"92d43f655ef348a692831c5af08bfd4a","collectionName":"torch.nn.SiLU","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.SmoothL1Loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f48aa565648f404bb3e3f8bb49902135","collectionName":"torch.nn.SmoothL1Loss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.SoftMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eaac770374a94dde9f326071600532ca","collectionName":"torch.nn.SoftMarginLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cd7b2430f79749c7b84cc27171a0b278","collectionName":"torch.nn.Softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softmax2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4d2b393327784c00bf5831a5558d7bb7","collectionName":"torch.nn.Softmax2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f222ab25ff2b4f14b862f3512ab6cc6e","collectionName":"torch.nn.Softmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softplus.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1abdd410876544ca96e6e93a4aaf0b59","collectionName":"torch.nn.Softplus","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f59a9807656c415198127ee6e3d7227b","collectionName":"torch.nn.Softshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Softsign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e42369632c7048babe7410517c881612","collectionName":"torch.nn.Softsign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9d11f09c1cbf461ba50aff47ca5d36b6","collectionName":"torch.nn.Tanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Tanhshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1def15d45a1b43b3b8e08c8202989233","collectionName":"torch.nn.Tanhshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Threshold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d6632bb33f84458b184aff321898b93","collectionName":"torch.nn.Threshold","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Transformer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5fcd416f9e99487fb0c4bb857426906a","collectionName":"torch.nn.Transformer","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.TransformerDecoderLayer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"94e3913f525e437e9237fdbfcdaff9a7","collectionName":"torch.nn.TransformerDecoderLayer","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.TransformerEncoderLayer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"84ec9a74ff5444e88035db96e88492fe","collectionName":"torch.nn.TransformerEncoderLayer","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.TripletMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6a8f2bbd708548029cebef708d5c7a9b","collectionName":"torch.nn.TripletMarginLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.TripletMarginWithDistanceLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ed0422da8704d499b60341bc5291998","collectionName":"torch.nn.TripletMarginWithDistanceLoss","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.Upsample.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f0952e8f54f64af1b18264a16d12b3fd","collectionName":"torch.nn.Upsample","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.UpsamplingBilinear2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a61a30cabcb440a7b93f810bebbab6f1","collectionName":"torch.nn.UpsamplingBilinear2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.UpsamplingNearest2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"10f537556c0d4228843878a9b100b725","collectionName":"torch.nn.UpsamplingNearest2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.utils.clip_grad_norm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c39d90a0f1e449e2899974377549178d","collectionName":"torch.nn.utils.clip_grad_norm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.utils.clip_grad_value_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c748b116c5a1458b9ed990412ba8c808","collectionName":"torch.nn.utils.clip_grad_value_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.utils.parameters_to_vector.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95f29f8a1ecc4db98f847dd4ea893ff3","collectionName":"torch.nn.utils.parameters_to_vector","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.utils.rnn.pack_sequence.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"abc457fd5e634615b6711460e6c732a3","collectionName":"torch.nn.utils.rnn.pack_sequence","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.utils.rnn.pad_sequence.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ec2dcafeb61440a81fa36722a713fef","collectionName":"torch.nn.utils.rnn.pad_sequence","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nn.ZeroPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bcabb130f6d6404c9257c653510bd230","collectionName":"torch.nn.ZeroPad2d","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9baaa477d2f947b789398237047406dc","collectionName":"torch.nonzero","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a6be17a4ed7e49c4be75a069d3098843","collectionName":"torch.norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.normal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"449b11a2ed534efeb0072d576e9742c2","collectionName":"torch.normal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.not_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"30dfa1627b0f4e84844ff4e987efe07c","collectionName":"torch.not_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.numel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"11c48ef0698047f2956626a48bf64e2e","collectionName":"torch.numel","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ones.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"226e7825c68b490886281772b0af7e38","collectionName":"torch.ones","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ones_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fb8d39b052c54faaa0b9be81205f7085","collectionName":"torch.ones_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.outer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56e1a9edd87b4416ae9e31845592b157","collectionName":"torch.outer","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.overrides.is_tensor_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cee183ecddbb4df88fda99e642c49430","collectionName":"torch.overrides.is_tensor_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.overrides.wrap_torch_function.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c44de653d0464f08a5dc381acb16195a","collectionName":"torch.overrides.wrap_torch_function","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.pca_lowrank.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"502ec08c1122476b8738e5dd958d3f80","collectionName":"torch.pca_lowrank","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.pinverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"093315ef6993424580461b584539d1c6","collectionName":"torch.pinverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.poisson.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57edfe081a9c474591f64cd65dad73d4","collectionName":"torch.poisson","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.polar.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17a67882363d4d1fabae270a49370e5b","collectionName":"torch.polar","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.polygamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6fbcace0e04f4f65b4162cf1526243ad","collectionName":"torch.polygamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.positive.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f5291aff2fe24708843eb68c24e5dae9","collectionName":"torch.positive","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.pow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"31bd077fa80e4e40966ce462ccd67e6f","collectionName":"torch.pow","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.prod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b15548f76d15461a9ec47f90427ccc7c","collectionName":"torch.prod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.promote_types.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a1c88d0fdf1497480c89cd2d534e2aa","collectionName":"torch.promote_types","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.qr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8acb420b3af645189d91462199e7ccbf","collectionName":"torch.qr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.quantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"31727bd2ac7f463295062dac65b16fbb","collectionName":"torch.quantile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.rad2deg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fbcee919eced46e9ae9c411207940a12","collectionName":"torch.rad2deg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.rand.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3df61c9745864258b07a2e057b01f415","collectionName":"torch.rand","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.randint.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d4e53faefae4623884d8501352f5cef","collectionName":"torch.randint","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.randn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f3feb544be9f44bbb57f2860df96095d","collectionName":"torch.randn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.randn_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"00c87987acc24512b19144310b419b55","collectionName":"torch.randn_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.randperm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e260647e0ad443ab91ef452f2f0e36f1","collectionName":"torch.randperm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.rand_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3e0673dcf8e54c06a25ffae0fdb992ff","collectionName":"torch.rand_like","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.range.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c0f47f9e79245d4962f87b941b1b347","collectionName":"torch.range","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.ravel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"74f9369854c5412da1637eb2ef158cd7","collectionName":"torch.ravel","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.real.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8c146a994564eb4b589d3925b9666ec","collectionName":"torch.real","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.reciprocal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"59cf0e5427b64c0493d73461968bb625","collectionName":"torch.reciprocal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.remainder.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7d09353378c84b14ba5a8db28905533e","collectionName":"torch.remainder","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.renorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4af99900239a4f71bec5fe41b63655e0","collectionName":"torch.renorm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.repeat_interleave.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4eb25630b5794f79a31c1c09c9753031","collectionName":"torch.repeat_interleave","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.reshape.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"856dc8379893477b914500db83541d9b","collectionName":"torch.reshape","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.resolve_conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"639f36eb420c40008b7c258ca99aac42","collectionName":"torch.resolve_conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.resolve_neg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d0678f160294e4692fc5e73cd30cc43","collectionName":"torch.resolve_neg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.result_type.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fbd927bf23a548b3834de708de4043ab","collectionName":"torch.result_type","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.roll.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1fa7235078a049eea20ab0c00b2dfb84","collectionName":"torch.roll","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.rot90.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a9855f51ea5848d49ec11253db613245","collectionName":"torch.rot90","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.round.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"741e1a7a181b480684b8063233098053","collectionName":"torch.round","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.rsqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e27c4850e0a44821916c351d4565d5e3","collectionName":"torch.rsqrt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.save.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"37eb5dbfae8e4500bef97463a4231da3","collectionName":"torch.save","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.scatter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"850e12af68a4489bb4702006c6025dd1","collectionName":"torch.scatter","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.scatter_add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e45f0365ac354e8b8fb1f62aee021190","collectionName":"torch.scatter_add","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.searchsorted.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"847ba0eeef334983a8174343ed41f2fd","collectionName":"torch.searchsorted","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.set_default_dtype.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0178b20fdb6945a5bdfb3502124ebaf0","collectionName":"torch.set_default_dtype","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.set_flush_denormal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f0ffa7f654e34ce495793ba6e5b5465e","collectionName":"torch.set_flush_denormal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.set_num_interop_threads.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"572361fe34d7427c948cd802a8cb6102","collectionName":"torch.set_num_interop_threads","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.set_num_threads.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9c9c0d2a138240e58b0e71bdadffa544","collectionName":"torch.set_num_threads","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.set_warn_always.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3ad8700023314fa787a2238ee9399d18","collectionName":"torch.set_warn_always","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sgn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"385c684f2b42438f88adc249d3da775c","collectionName":"torch.sgn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"32f1f4eba5df420a89ef29f3bc0ce2fe","collectionName":"torch.sigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b8bfc96f8a9345b49cf0515b74e75343","collectionName":"torch.sign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.signbit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c388ff5a32e2412aafc6b6eed7d66514","collectionName":"torch.signbit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de739e158b584523bf3912829dad5fab","collectionName":"torch.sin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sinc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"01db39db244c49efa3180633b1026dec","collectionName":"torch.sinc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e65a77d1b46476dbac36eb1499d50dd","collectionName":"torch.sinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.slogdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67dafbe8cc244c0dbdce3b05af3522b1","collectionName":"torch.slogdet","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.smm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2cf4470f5235436e9a0f6816a0d02fd5","collectionName":"torch.smm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36369eb66aa84f81bdae65e77d1023cc","collectionName":"torch.solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac8ce0a40f334e64a929114ef2d44b5e","collectionName":"torch.sort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse.addmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"875483956f0141e1b6260d3f31734805","collectionName":"torch.sparse.addmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse.log_softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95d59a9876164164bd8845fcb55aaaf3","collectionName":"torch.sparse.log_softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse.mm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d887dfe7a6db4cb68b0181ba6290f36b","collectionName":"torch.sparse.mm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse.softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8d4d4b0d46f6422495c5a04d742c90fd","collectionName":"torch.sparse.softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse.sum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ab8791a36fe47448481dc54ded60d38","collectionName":"torch.sparse.sum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sparse_coo_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f17f930f69b43339b86ba2b166a6c91","collectionName":"torch.sparse_coo_tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.digamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a328170fbbfe46d5a8b5c7cfb011ea09","collectionName":"torch.special.digamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.entr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1f52ac553335467999570828b7027158","collectionName":"torch.special.entr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.erf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"16cbc9a7fa774fec8dd579d1b804d033","collectionName":"torch.special.erf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.erfc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fdbce10317254184aa03e437c54e0cb0","collectionName":"torch.special.erfc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.erfcx.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"70bc492b3d274730a3195cc3a097eab3","collectionName":"torch.special.erfcx","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.erfinv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"aa61d844055b4e278854af2393d1daba","collectionName":"torch.special.erfinv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.exp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"75566c5c438b44eaa3299142fdbcfa19","collectionName":"torch.special.exp2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.expit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b02cb74550224a288a506442da227921","collectionName":"torch.special.expit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.expm1.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc22aa8526294b60ae35d22e017bf3a9","collectionName":"torch.special.expm1","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.gammaln.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b404a3c65c9940d2bdc740c6f280993d","collectionName":"torch.special.gammaln","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.i0.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b0eb07e502bc4d9daf9c582e63243685","collectionName":"torch.special.i0","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.i0e.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cdd60b9805a948c786b0c5c923219172","collectionName":"torch.special.i0e","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.i1.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c97d34fcabed48b585f81b6dc5dda93e","collectionName":"torch.special.i1","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.i1e.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"202722f7c63f478386d212b3b8ed2233","collectionName":"torch.special.i1e","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.log1p.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dfc57f88d33b48e591b3b3b5ecd6e095","collectionName":"torch.special.log1p","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.logit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3242ffe1eee64df0993d5c8557d5e34d","collectionName":"torch.special.logit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.logsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d89fa6f3e6e640de889d5b042da3ab0e","collectionName":"torch.special.logsumexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.log_softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bec1903a82d14017bc6349339f9f8ea0","collectionName":"torch.special.log_softmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.multigammaln.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ef20da5a413473ca6d1a10125d00175","collectionName":"torch.special.multigammaln","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.polygamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"704afbc4aa594048b51ef2f48bcc5237","collectionName":"torch.special.polygamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.psi.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"41b7b095aeee444ca582e3b48b1762b0","collectionName":"torch.special.psi","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.round.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f689abc71e7d430ca199ca7f7211b461","collectionName":"torch.special.round","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.sinc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4da9649f331d44cd81f348e9d7d1f73d","collectionName":"torch.special.sinc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.xlog1py.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"99274276f1384226bad6fb360dc80868","collectionName":"torch.special.xlog1py","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.special.xlogy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c37c2b7b642049afa7b504aa6b9ed668","collectionName":"torch.special.xlogy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"00bf51eb8853489ca31582d09725ffe5","collectionName":"torch.split","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9f43eb251a84335a9d985402a984570","collectionName":"torch.sqrt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.square.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d96cf883bed8460bbc4e1e68a9157200","collectionName":"torch.square","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.squeeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e4d3f0d7b65943dd82f9db703e04ee49","collectionName":"torch.squeeze","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sspaddmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"198eebbfce3f48188266c0530023641d","collectionName":"torch.sspaddmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.stack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"972c5f9579b149f3825d72c179447006","collectionName":"torch.stack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.std.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b1006a65ad5f48f38b3da4f83fb21cc7","collectionName":"torch.std","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.std_mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5f814ef1fe946329a6f5d7dffbe0dba","collectionName":"torch.std_mean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sub.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b6f8b6adbc324ed7b194ccd8e4794abf","collectionName":"torch.sub","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.subtract.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"04d1e376e14942a8a02c8cbea6328806","collectionName":"torch.subtract","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.sum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"89e774f782914b36bb3ef4b2bdf4ac80","collectionName":"torch.sum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.svd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4d4aa03a6e624e5a844e0196a6cc9b93","collectionName":"torch.svd","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.svd_lowrank.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"925cd6447c694a15a60750e4ccc0c656","collectionName":"torch.svd_lowrank","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.swapaxes.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"992889ab41b7424794d8e758a8c467dc","collectionName":"torch.swapaxes","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.swapdims.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f7e65c5b62b640108121fd3c258b7951","collectionName":"torch.swapdims","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.symeig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bfdee9820c6b41c9af3b65c95e183b69","collectionName":"torch.symeig","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.t.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b653435e1374cfcb5ebf92f1af744db","collectionName":"torch.t","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.take.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5487c2c66a0c4a81ad046d11cb8d04f4","collectionName":"torch.take","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4fd6e0c691c445aa90f4b345ceaaf33d","collectionName":"torch.tan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b60bb5806d9b4da8a9777f5096d8f57f","collectionName":"torch.tanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.abs.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"580496dc5b0940429426265a7fde82d4","collectionName":"torch.Tensor.abs","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.absolute.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2070ebab78744fd7825f79e4540b289a","collectionName":"torch.Tensor.absolute","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.absolute_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"70f1b3cf74b64ce887afd33b608d23ca","collectionName":"torch.Tensor.absolute_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.abs_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"868f968bb781483dbc8ce5b11bb53f0f","collectionName":"torch.Tensor.abs_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.acos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e6269b1c1a94380bedf795cc443ce2e","collectionName":"torch.Tensor.acos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.acosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ed7983c68b8143b9a3050f00b9fcd14a","collectionName":"torch.Tensor.acosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.acosh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b126e6dac6b4df791207a0ee71856c3","collectionName":"torch.Tensor.acosh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.acos_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"aa8b00996f1e43dd91be35ac79328d03","collectionName":"torch.Tensor.acos_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9c308f03a7f44add8c46371fabaacca4","collectionName":"torch.Tensor.add","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ed50247bb0d543c381451d0f3af4f6be","collectionName":"torch.Tensor.addbmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addbmm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0a3178e56fd54ac0b80cb470e8deb12a","collectionName":"torch.Tensor.addbmm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addcdiv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d249af3449744b75bf547c885eae9a06","collectionName":"torch.Tensor.addcdiv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addcdiv_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5c1baef38574b54b5aaf189288e7490","collectionName":"torch.Tensor.addcdiv_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addcmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d5940dac6d149f386a21a39131ce918","collectionName":"torch.Tensor.addcmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addcmul_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a491e59f1a44441dbf06d324f2174bd8","collectionName":"torch.Tensor.addcmul_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ccdb3e32f6d4e6995af5ddf1dfbdac0","collectionName":"torch.Tensor.addmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addmm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bb00ea283ceb435194f719c7d724fb76","collectionName":"torch.Tensor.addmm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addmv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9736188dee6f4ebfaec41e7bde5991e7","collectionName":"torch.Tensor.addmv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addmv_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"722aa0ca168d483e8cbd9ca6940747e6","collectionName":"torch.Tensor.addmv_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"050c62ac14f44fd8b35f519e77b421b9","collectionName":"torch.Tensor.addr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.addr_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba83535f71384c0ab1b44e96d4828045","collectionName":"torch.Tensor.addr_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.add_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f39c53db8a3f401aa217ddbd254b6972","collectionName":"torch.Tensor.add_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.all.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e7c21ce747a341d39d3de5866e96032d","collectionName":"torch.Tensor.all","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.allclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eba7fd92a832432e868291cdd8f8c782","collectionName":"torch.Tensor.allclose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.amax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cc770c7600704417ac589d067396fd04","collectionName":"torch.Tensor.amax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.amin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3340383ad7bf421cacd2150cdb473b7c","collectionName":"torch.Tensor.amin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.aminmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"74b60014c01c4639a734f4e88911d336","collectionName":"torch.Tensor.aminmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.angle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b8078aa58e84c7ea591045256b440ae","collectionName":"torch.Tensor.angle","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.any.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b23ea1e4cfa446abba0a8c434b1a98c2","collectionName":"torch.Tensor.any","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arccos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c7e8dbf7e138479684e55c5c2f09997a","collectionName":"torch.Tensor.arccos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arccosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8f237f3440d402d9c8331fb1b2965d9","collectionName":"torch.Tensor.arccosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arccos_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b31f2683ed924621883cfa919a570625","collectionName":"torch.Tensor.arccos_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arcsin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"365ed503e2b94fcab61589e95d222574","collectionName":"torch.Tensor.arcsin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arcsinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db6e7068df36477cbb0b748eabe4df30","collectionName":"torch.Tensor.arcsinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arcsinh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67208b45391144c6ad25c3f73aa29d36","collectionName":"torch.Tensor.arcsinh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arcsin_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"34189349aabb47aa9df59d9223b30632","collectionName":"torch.Tensor.arcsin_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arctan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f4ac403e42343b2bcfa6c437996f5c0","collectionName":"torch.Tensor.arctan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arctanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cb732adcea9c4a52800d92b4312ebc64","collectionName":"torch.Tensor.arctanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arctanh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7534038a75564353ab775b4e998f688f","collectionName":"torch.Tensor.arctanh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.arctan_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0401da40899847b78e768cf7c7eb8152","collectionName":"torch.Tensor.arctan_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.argmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"97b107c2968943a8867d4372cc7ad8a4","collectionName":"torch.Tensor.argmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.argmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"da861666996649c791dd3d8c7f770b95","collectionName":"torch.Tensor.argmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.argsort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5dcb61d3159a4144b67afc331b7570e1","collectionName":"torch.Tensor.argsort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.asin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b5790f431184127ac440d550dd581db","collectionName":"torch.Tensor.asin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.asinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f392932dca1f44eea19300ac478167ff","collectionName":"torch.Tensor.asinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.asinh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"efe9a358c03a49e2a9464e34fc286aae","collectionName":"torch.Tensor.asinh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.asin_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de2950830345423c8d48870020108258","collectionName":"torch.Tensor.asin_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.as_strided.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3445c66ef21048fd8310b04aaa1fb9c3","collectionName":"torch.Tensor.as_strided","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"059c430abf5a4522b4b06bcafe79a3ba","collectionName":"torch.Tensor.atan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atan2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4f44a585e1ad4766a90e7c7e0af03b11","collectionName":"torch.Tensor.atan2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atan2_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ebad0947a6a64aeab6a48a2704df38e9","collectionName":"torch.Tensor.atan2_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c65ceb94a024a60b15277ec3db56c3e","collectionName":"torch.Tensor.atanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atanh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5d4ff72ab946434494f46a2240333c8b","collectionName":"torch.Tensor.atanh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.atan_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f33806697a174371b7c7b61927a664d6","collectionName":"torch.Tensor.atan_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.baddbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f47d883f31ca4336a6b50e36b1e81bd0","collectionName":"torch.Tensor.baddbmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.baddbmm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ce554a172774b6390ef8f9239660458","collectionName":"torch.Tensor.baddbmm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bernoulli.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"27300b135ad94978b344fe9ff58400d1","collectionName":"torch.Tensor.bernoulli","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bernoulli_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"89c2001d1e4e45b8ad97911aa95e383c","collectionName":"torch.Tensor.bernoulli_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bfloat16.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d40d38e552f14cf1a166b94a7d962190","collectionName":"torch.Tensor.bfloat16","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bincount.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9aef7aebec814f07bf88470f808b2570","collectionName":"torch.Tensor.bincount","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a8e10947fe84814b5e0f42abe21f482","collectionName":"torch.Tensor.bitwise_and","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_and_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fd7ac0f19f384e1ca44b85a6983f36ea","collectionName":"torch.Tensor.bitwise_and_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_left_shift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c7e84e78c154e648acd10ae29f85fb2","collectionName":"torch.Tensor.bitwise_left_shift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_left_shift_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a56e4b708b904cc39552db6652b2a034","collectionName":"torch.Tensor.bitwise_left_shift_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ac1129458534c34b6b0476a51ad6e2b","collectionName":"torch.Tensor.bitwise_not","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_not_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"30a1f0096ad44567a999150fb4de6cd7","collectionName":"torch.Tensor.bitwise_not_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"af2c733adcc34f0c8dc94a5ad3ccc97d","collectionName":"torch.Tensor.bitwise_or","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_or_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b5f7859968f41b5af6c80331eeb02c4","collectionName":"torch.Tensor.bitwise_or_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_right_shift.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2810e5ab14714eb5858ad3f12295fda7","collectionName":"torch.Tensor.bitwise_right_shift","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_right_shift_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2632cb61786d417693aef4f73cdcfdb4","collectionName":"torch.Tensor.bitwise_right_shift_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fbf47510cec94afc97c477c87517249b","collectionName":"torch.Tensor.bitwise_xor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bitwise_xor_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"48b2f64cb137424aae296165aa75766a","collectionName":"torch.Tensor.bitwise_xor_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b0b7e029774f4a4d81f75792a0cd874f","collectionName":"torch.Tensor.bmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.bool.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"234ee45569514a159e43f3e765c40b2a","collectionName":"torch.Tensor.bool","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.broadcast_to.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67c151dcd417452c98ed95f2ea50c9f7","collectionName":"torch.Tensor.broadcast_to","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.byte.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"92a7aa8a201a42a5a5faabb02d7dbda4","collectionName":"torch.Tensor.byte","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cauchy_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"26df769f65a843f7940a73e9e7595e91","collectionName":"torch.Tensor.cauchy_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ceil.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8a6581b4b9db4ad7bd295058a8b3683b","collectionName":"torch.Tensor.ceil","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ceil_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1ffbba5e4c7740f6a908a8e33b568208","collectionName":"torch.Tensor.ceil_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.char.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8c5ce0b0f6c947f9bbbbb0ac85290e0c","collectionName":"torch.Tensor.char","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cholesky.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"673c42501d074d498b9f2cbcc31d5ade","collectionName":"torch.Tensor.cholesky","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cholesky_inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"369d7c7015d4411392bf8854e86aa26f","collectionName":"torch.Tensor.cholesky_inverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cholesky_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52290d7f35d744bc9d1591aaf8c14a2b","collectionName":"torch.Tensor.cholesky_solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.chunk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5fd42172682d4d4bbba16c81f2bcaebf","collectionName":"torch.Tensor.chunk","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.clamp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"827db69bcd2f49f3b10d15c5714929a0","collectionName":"torch.Tensor.clamp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.clamp_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bc1cd7dd9a294f808beae442cc6d882e","collectionName":"torch.Tensor.clamp_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.clip.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"49aa789700774327abff5944f088b1e8","collectionName":"torch.Tensor.clip","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.clip_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b21e88b029364a4b8fb583d6d85c7a21","collectionName":"torch.Tensor.clip_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.clone.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b4972b126b354ca9a4b1ac28981c7c47","collectionName":"torch.Tensor.clone","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"42a9b795a9d34c4baa9a012308052dc3","collectionName":"torch.Tensor.conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.conj_physical.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2d055bf42abc441ea6120a9f4e994723","collectionName":"torch.Tensor.conj_physical","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.conj_physical_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5c8f77f5e7404d4aa338d16f0eb04023","collectionName":"torch.Tensor.conj_physical_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.contiguous.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f21d8e8688bb4b7ba606d6bb02009e40","collectionName":"torch.Tensor.contiguous","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.copysign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"74b1e567905641f892dcc269c2e83909","collectionName":"torch.Tensor.copysign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.copysign_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6fb1a18f989146f9a16f2fed57c58d04","collectionName":"torch.Tensor.copysign_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.corrcoef.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"796be80e6fa644edb1a115b99981fb55","collectionName":"torch.Tensor.corrcoef","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5f84d6efa4864bdcac5f3c2c78336380","collectionName":"torch.Tensor.cos","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e67fa01a6a8d456b9138c333a3e4b5aa","collectionName":"torch.Tensor.cosh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cosh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"518658613bcc4e40b60c0fa9e38db40a","collectionName":"torch.Tensor.cosh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cos_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4d4a5384d36846e8bf5a9f52d6f90b48","collectionName":"torch.Tensor.cos_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.count_nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"edd77489afdb422ea5b8aaba439c1d25","collectionName":"torch.Tensor.count_nonzero","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cov.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8a6cbc2b35304aa19e6941e809e2ef6f","collectionName":"torch.Tensor.cov","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cpu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eefa4cde36244fd480df75ba6836ab07","collectionName":"torch.Tensor.cpu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cross.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2a38d57303084236b56893d680dbe0b7","collectionName":"torch.Tensor.cross","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cummax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"70ec88585f2b448586a20ea85b2809aa","collectionName":"torch.Tensor.cummax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cummin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e59c89202a3f45278c53c987b8233f8f","collectionName":"torch.Tensor.cummin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cumprod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"654203575d6849958d1e17574170b255","collectionName":"torch.Tensor.cumprod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cumprod_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"41f66cb47870414cb473ea99c923faab","collectionName":"torch.Tensor.cumprod_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cumsum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8b31495a3904039a4d1c83390b06178","collectionName":"torch.Tensor.cumsum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.cumsum_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"971206fc088a4696bcb20742f61fd9ac","collectionName":"torch.Tensor.cumsum_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.data_ptr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"acceb08d622f4b8098743fc40eeb4600","collectionName":"torch.Tensor.data_ptr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.deg2rad.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f2211cf0129f49c39f0c8e08d62d21ff","collectionName":"torch.Tensor.deg2rad","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.det.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2fbc9b2e7cff45568c7cb8456368e800","collectionName":"torch.Tensor.det","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.detach.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"38708b7ebfa143128bc194a0f1e1fc4b","collectionName":"torch.Tensor.detach","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.detach_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ed5b701dddf4bb98fbf934acfc6822c","collectionName":"torch.Tensor.detach_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.diag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c8f4e735e58a4b6989deeefadc604be5","collectionName":"torch.Tensor.diag","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.diagflat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d511496f15044b43a057c43b5dcb407e","collectionName":"torch.Tensor.diagflat","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.diagonal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"82db1fd4b12046cd9ba1bb68ce35ef83","collectionName":"torch.Tensor.diagonal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.diag_embed.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a24d802f2e543d69b93363c73f50520","collectionName":"torch.Tensor.diag_embed","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.diff.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b50610bc5324715b72a589a4271829f","collectionName":"torch.Tensor.diff","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.digamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8cbb62a464a1470b83b558dd7d70351e","collectionName":"torch.Tensor.digamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.digamma_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"02c693bba5c54a7bb9fcae7a29a48e3b","collectionName":"torch.Tensor.digamma_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.dim.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"087ce0b150914d328517ce1fe1501ee8","collectionName":"torch.Tensor.dim","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.dist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8877610eb8a544d1887805dc5cd51c16","collectionName":"torch.Tensor.dist","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.div.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7adfb62005254cc0b30a03fde29ca35c","collectionName":"torch.Tensor.div","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2afd451bc96b4ae1844c5efa678aca25","collectionName":"torch.Tensor.divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.divide_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d63f9bd9a21649a48454ccbc4c27bd3e","collectionName":"torch.Tensor.divide_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.div_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a3e43963ab964babb55225f3ef6047f1","collectionName":"torch.Tensor.div_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.dot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"29d7478387ef4060a3836757bbb1e849","collectionName":"torch.Tensor.dot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.double.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"46a0198bf7c54ceb8409cd1a056c89b9","collectionName":"torch.Tensor.double","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.dsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7b624d73d8a74cc594661e970eb275b5","collectionName":"torch.Tensor.dsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.eig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0997963936334caf9fb5ab0a62c6f090","collectionName":"torch.Tensor.eig","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.element_size.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ee9c55ab7644675b116ea1d1f6f055f","collectionName":"torch.Tensor.element_size","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.eq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e822b9d2bc0843558c6d038feb18ad9d","collectionName":"torch.Tensor.eq","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"df7e9673d9e64b62b7d7f400f87f79d1","collectionName":"torch.Tensor.equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.eq_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c8c05b2742443409711bcf38da160fb","collectionName":"torch.Tensor.eq_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"993c9edaf3fb4e679c23500ba9601f46","collectionName":"torch.Tensor.erf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erfc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c0cd02da337249758234cc5e97c11ecf","collectionName":"torch.Tensor.erfc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erfc_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dbd6a0d95aba4d45a37c0c24ec4310ee","collectionName":"torch.Tensor.erfc_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erfinv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c3ef7b9e43914b29b304981a9570ce3e","collectionName":"torch.Tensor.erfinv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erfinv_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4705320f3d09494abf004b921cc9374d","collectionName":"torch.Tensor.erfinv_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.erf_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b2ea4363bdac4c1c82ea32ad47a245a4","collectionName":"torch.Tensor.erf_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"24aea094a8a949bd9326bcadcc30a20c","collectionName":"torch.Tensor.exp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.expand_as.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b325a630ea34fc4a724e69c8a395f44","collectionName":"torch.Tensor.expand_as","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.expm1.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"784f60d2e21e49ccbbcccd6b77ca78b1","collectionName":"torch.Tensor.expm1","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.expm1_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cde936f5d703462aad787045be74ee80","collectionName":"torch.Tensor.expm1_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.exponential_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c93dc53541a34ad78d744fb0868721cd","collectionName":"torch.Tensor.exponential_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.exp_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d0bfbbc4d0494a0589e9450c79df38c7","collectionName":"torch.Tensor.exp_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fill_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a3b15d845b3747ddadd1e170c96a50df","collectionName":"torch.Tensor.fill_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fill_diagonal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b46c3dadc94427d94cfe07e8ae87267","collectionName":"torch.Tensor.fill_diagonal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fix.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"78c4c3de979944d5bf7f6fb614d64c1b","collectionName":"torch.Tensor.fix","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fix_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"056dc428cfda42fab3465b79e6101ade","collectionName":"torch.Tensor.fix_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.flatten.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b24b9a1418964a0289a3f160b51fcaa8","collectionName":"torch.Tensor.flatten","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.flip.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"14ff9718c67149f09012327448986652","collectionName":"torch.Tensor.flip","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fliplr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"34a417a30ad241119ac1cbfb84706165","collectionName":"torch.Tensor.fliplr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.flipud.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9960259180a045c9a44635be15168be6","collectionName":"torch.Tensor.flipud","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.float.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c1b4add753c246789cf9cb968c02d06d","collectionName":"torch.Tensor.float","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.float_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8d644cff0ad430cb63e494e1303a286","collectionName":"torch.Tensor.float_power","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.float_power_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52be776973cf4854a8e8364fc180d627","collectionName":"torch.Tensor.float_power_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.floor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f19b8edead34497dbf0757916482020f","collectionName":"torch.Tensor.floor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.floor_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"afd2f22c3b5d41e88723138be85faa52","collectionName":"torch.Tensor.floor_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.floor_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d455ee9ce41749ad9f6bed730708192b","collectionName":"torch.Tensor.floor_divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.floor_divide_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0c01eff1f61a4e9486166b339160fdde","collectionName":"torch.Tensor.floor_divide_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b97408f22694ec4aa88e200afad883e","collectionName":"torch.Tensor.fmax","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"045e57d0d2d94daea121b1c3731bf0df","collectionName":"torch.Tensor.fmin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fmod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"21bb6b23070743a2994afb1665a80d6a","collectionName":"torch.Tensor.fmod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.fmod_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67e6a847bd7c44b6bcd25c1709f4f6c2","collectionName":"torch.Tensor.fmod_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.frac.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57bad1eb4fd04e1f980b462db7e4a4fd","collectionName":"torch.Tensor.frac","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.frac_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a43502fc5189483aa18b6d3544db1586","collectionName":"torch.Tensor.frac_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.gather.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96be2cd59ba648309df7904509349c59","collectionName":"torch.Tensor.gather","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.gcd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7e4f74b43f4f4186a7cabd3c48e9699c","collectionName":"torch.Tensor.gcd","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.gcd_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb06f37e549447fc85ae3fb54ee64730","collectionName":"torch.Tensor.gcd_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ge.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a097e0a1d82448cdb6a613b3d96e7daf","collectionName":"torch.Tensor.ge","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ger.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bdd6dee2b5b74148bdd564bd14ea6fc1","collectionName":"torch.Tensor.ger","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.get_device.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a8ada14669be4faeaa6253b0e0a8e678","collectionName":"torch.Tensor.get_device","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ge_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a77b00627e54cd1a8590a475acccd31","collectionName":"torch.Tensor.ge_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.greater.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1fe960773f494934b3d44c92390de48f","collectionName":"torch.Tensor.greater","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.greater_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17d0eaea01fd4269922a0a78459dbf64","collectionName":"torch.Tensor.greater_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.greater_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d92bc7ddebd74556b2606d0ae027567d","collectionName":"torch.Tensor.greater_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.greater_equal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"39d2ee48c5eb47cf955d4beebaa62506","collectionName":"torch.Tensor.greater_equal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.gt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c10d9f23eecc4c44b22360a1ad52eee7","collectionName":"torch.Tensor.gt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.gt_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8c0c3ed9f35e4d17856711fb7b1f5a7e","collectionName":"torch.Tensor.gt_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.half.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b9a3d1d75b34b2ca5fe12b635f66896","collectionName":"torch.Tensor.half","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.hardshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"03f3d47986824d13982d1ac11bd92eca","collectionName":"torch.Tensor.hardshrink","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.heaviside.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"be8d39de45894fd7a169d92647b81cda","collectionName":"torch.Tensor.heaviside","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.histc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ea4689fcf7424bb08f20500efb20f1f4","collectionName":"torch.Tensor.histc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.hsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"527a2aa8a6d04375ab768faf01aa14fe","collectionName":"torch.Tensor.hsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.hypot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"82b201bab53846c5a878ec740e601d1f","collectionName":"torch.Tensor.hypot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.hypot_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dc8a4a96ba944a4a850ccd9fbeb4a915","collectionName":"torch.Tensor.hypot_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.i0.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc6590382af34b8bbd7f3c729c70bf67","collectionName":"torch.Tensor.i0","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.i0_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2078b044fd104ef38550b081508ca21d","collectionName":"torch.Tensor.i0_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.igamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c42f32b46e85425bbe7874d8d7f660d6","collectionName":"torch.Tensor.igamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.igammac.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0e7200100a774eebb1a1ab3a1d84035d","collectionName":"torch.Tensor.igammac","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.igammac_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f1bee166dc4245f0b798b760683dcfd9","collectionName":"torch.Tensor.igammac_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.igamma_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"69ebbbe9df504fb693445ed7139797e3","collectionName":"torch.Tensor.igamma_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"16da95782b7745ed917503634e35fdde","collectionName":"torch.Tensor.index_add","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_add_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"410b84901aaa4052ac42004f2353df4e","collectionName":"torch.Tensor.index_add_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_copy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4aa629e7e4b54334bfebfd6a8a52d434","collectionName":"torch.Tensor.index_copy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_copy_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e950be6daf654b919a776557e565f58c","collectionName":"torch.Tensor.index_copy_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_fill.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"471e8b61196d4259b8546c3cab3481fa","collectionName":"torch.Tensor.index_fill","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_fill_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d3600e9a523402c88311ef0146292da","collectionName":"torch.Tensor.index_fill_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.index_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bc3b82a7f93649d2a755b215c8319d5a","collectionName":"torch.Tensor.index_select","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.inner.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"228f25c6895b4dfca4da529392aed729","collectionName":"torch.Tensor.inner","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.int.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d141ff1ec305484582e4d373569452fd","collectionName":"torch.Tensor.int","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"35316a1bb3f8446baebd57b425cfc73b","collectionName":"torch.Tensor.inverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23c68d9941ff4e5ca636e1860b4f5481","collectionName":"torch.Tensor.isclose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isfinite.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1c57c7d8205a4090b937ae7d04ae6565","collectionName":"torch.Tensor.isfinite","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"769c9cce98864e44a6f3ea7b32ff7bc7","collectionName":"torch.Tensor.isinf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isnan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9c8ab9b75d664fedb58fe0076dd82fd6","collectionName":"torch.Tensor.isnan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isneginf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"909c9cee53fa45188c77e3e4a2d00260","collectionName":"torch.Tensor.isneginf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isposinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bdbee6648f024e85914157410748cbf7","collectionName":"torch.Tensor.isposinf","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.isreal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"516d9acae8644d209f21b4a7f2e66ff9","collectionName":"torch.Tensor.isreal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c0752bbc7c544166bb3b6a8da1cf34e2","collectionName":"torch.Tensor.is_complex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a6d65f7b29bd48d7ae44285254a5fde7","collectionName":"torch.Tensor.is_conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_contiguous.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"601c66b35ade41328f6dc0394c0a9d9e","collectionName":"torch.Tensor.is_contiguous","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_floating_point.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c6c3dffb883b40949ced38416a0f24f9","collectionName":"torch.Tensor.is_floating_point","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_inference.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"27b7e99f9e2a433ca16f992fc3d7b384","collectionName":"torch.Tensor.is_inference","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_pinned.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8a15760054654e73ae3189e4e6ff9d8d","collectionName":"torch.Tensor.is_pinned","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_set_to.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"03421dba82f3432392773da1686bd3d0","collectionName":"torch.Tensor.is_set_to","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_shared.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3fbc912ff8a34d0eb47a7f99604911db","collectionName":"torch.Tensor.is_shared","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.is_signed.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"af9aee1d8c814758b77d0b9cbd26d8e1","collectionName":"torch.Tensor.is_signed","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.kthvalue.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b0f565712334353935e6d7616f4c615","collectionName":"torch.Tensor.kthvalue","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lcm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9751864d2c24307bf37c3aa4fdd782e","collectionName":"torch.Tensor.lcm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lcm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b34b05a14334d13bf77dbb0f62d232d","collectionName":"torch.Tensor.lcm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ldexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b6515ffbf1d441fcb774e4b53d43b310","collectionName":"torch.Tensor.ldexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ldexp_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5ae6028cfef949f4906de87169fb4408","collectionName":"torch.Tensor.ldexp_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.le.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8dbac5396bdc4768b7bb4ecedd04c6db","collectionName":"torch.Tensor.le","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lerp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"45b4d6fc80984d05b4ffd955c72c9414","collectionName":"torch.Tensor.lerp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lerp_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5101e964e47946868bca9bbe71caa6ee","collectionName":"torch.Tensor.lerp_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.less.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4942382c26cf4809b20ff7b6e53bb322","collectionName":"torch.Tensor.less","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.less_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"155d439790f842c5ba24401d4e57e25f","collectionName":"torch.Tensor.less_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.less_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8d0daf64109b40f29bd8e1dd40c27aa4","collectionName":"torch.Tensor.less_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.less_equal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ceb5241e00c64fd59358b8afc5f2c61f","collectionName":"torch.Tensor.less_equal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.le_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f957200605684821b763456950c09c2f","collectionName":"torch.Tensor.le_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"50f37c7d59f74ac89dbaba483f7cc47d","collectionName":"torch.Tensor.lgamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lgamma_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0efa7ff888304035a9e297574f634abf","collectionName":"torch.Tensor.lgamma_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4ae4650a480f417397bc5a02fd767123","collectionName":"torch.Tensor.log","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log10.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"edcbc49c8ec2439da9651452d000b369","collectionName":"torch.Tensor.log10","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log10_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b8acfcf1cee34a41af0adcf195fad58c","collectionName":"torch.Tensor.log10_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log1p.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23d2ac0d6cc948a4a9cfad348dead3bd","collectionName":"torch.Tensor.log1p","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log1p_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a553f447d6f64cd7beeae0cf64367976","collectionName":"torch.Tensor.log1p_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"05789c4749714c4fbe342220d7ce4015","collectionName":"torch.Tensor.log2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log2_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"977ca5dff4f44831bd847d0e3547338a","collectionName":"torch.Tensor.log2_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logaddexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"63f7dc3340be433891b6f33f6b111bc5","collectionName":"torch.Tensor.logaddexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logaddexp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b6b0c47dd0dd4af8ad6fef1df9da10e0","collectionName":"torch.Tensor.logaddexp2","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logcumsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"499b71fa35544e4983cf61b70424d9b2","collectionName":"torch.Tensor.logcumsumexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7d8b92b119d740c7bc5d86f2f7ae3114","collectionName":"torch.Tensor.logdet","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe52cebdfcf14523aeccaeb0f656ef49","collectionName":"torch.Tensor.logical_and","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_and_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fb9839d707f84ed19779bfc08316f527","collectionName":"torch.Tensor.logical_and_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6feb5c4693bc4f34be4647449131693d","collectionName":"torch.Tensor.logical_not","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_not_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6ac5b9c7adde4d3f8037bc5bcad8282a","collectionName":"torch.Tensor.logical_not_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d22a4f0175f64bc9829ee7386afaa7cf","collectionName":"torch.Tensor.logical_or","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_or_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac4d62ec98864bd38052d6e8721544f3","collectionName":"torch.Tensor.logical_or_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac1451f6fc014be19da5c3491bb67d89","collectionName":"torch.Tensor.logical_xor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logical_xor_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bca7b5f4890e48bdb8094af1b4ee15e8","collectionName":"torch.Tensor.logical_xor_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bd11cf92d2cb4d86b30585d8d0c2bcb2","collectionName":"torch.Tensor.logit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logit_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"247baaca3ea84523898a4843a5abd893","collectionName":"torch.Tensor.logit_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.logsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b5279c930fc7490eb212892745190b2a","collectionName":"torch.Tensor.logsumexp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.log_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"88d720b72e2f481fad656e22bdf47b6d","collectionName":"torch.Tensor.log_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.long.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bcd2271753bd49849245c8615998e540","collectionName":"torch.Tensor.long","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lstsq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0917ce6af2ce45a7a7ef9996630e7519","collectionName":"torch.Tensor.lstsq","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96225f634a934abc876d92166c56cf97","collectionName":"torch.Tensor.lt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lt_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b6e2a6dc475d4f17bc59a60adc8d6e47","collectionName":"torch.Tensor.lt_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.lu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1aba80369b3e41628fa29d12da08ba97","collectionName":"torch.Tensor.lu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.masked_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e44f30fe424643d7a82e4a98cd55b7f8","collectionName":"torch.Tensor.masked_select","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7d1adef9768a4c6997f978e67c460fcd","collectionName":"torch.Tensor.matmul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.matrix_exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ab9c17bf3d7e494192685897c71101de","collectionName":"torch.Tensor.matrix_exp","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.matrix_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1892538f0848480f940d1d08e2c85461","collectionName":"torch.Tensor.matrix_power","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.max.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e52e6d3c0add4ee1847638f898efa606","collectionName":"torch.Tensor.max","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.maximum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a5b3aede73a4bf5a3416277e14bb732","collectionName":"torch.Tensor.maximum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"de631d0c43cf42cf92595e1f882138fe","collectionName":"torch.Tensor.mean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.median.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad4ed89e03cf494990e856bebb9d7b15","collectionName":"torch.Tensor.median","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba58de245f31479ebbeaa3cbbdb09b36","collectionName":"torch.tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.min.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f00317d400645baaabb3c1b64fdb19d","collectionName":"torch.Tensor.min","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.minimum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a5277e3a249c4552904d63d286b05dca","collectionName":"torch.Tensor.minimum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c198cf6671154df29b907dfc9e0b3d36","collectionName":"torch.Tensor.mm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mode.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9147042030004730bd4de8bff5ac0d28","collectionName":"torch.Tensor.mode","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.moveaxis.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b2a79290239b48e6a7b9e021f445c7b8","collectionName":"torch.Tensor.moveaxis","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.movedim.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e1dbd71c20654f969897a0631f53c7f5","collectionName":"torch.Tensor.movedim","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.msort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b3a5cb785bd9474eb90944d17875fa92","collectionName":"torch.Tensor.msort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4f67ba5730544b8b8cfdec404e8275da","collectionName":"torch.Tensor.mul","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.multinomial.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a86acd4801746b682b8699f89525daf","collectionName":"torch.Tensor.multinomial","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.multiply.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5099d7bc8b64e7ab107305888785327","collectionName":"torch.Tensor.multiply","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.multiply_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d48f440e12b4494ba9f7f967b3dd694","collectionName":"torch.Tensor.multiply_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mul_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ba781c46d9141a38d38bdc9a0e00b68","collectionName":"torch.Tensor.mul_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f2486f24c8444bdf82ba2df53c4eaf4a","collectionName":"torch.Tensor.mv","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mvlgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"383b8068f92748b0b9e29533ae04b810","collectionName":"torch.Tensor.mvlgamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.mvlgamma_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ee34498682b84722a92aa5829fda2c08","collectionName":"torch.Tensor.mvlgamma_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nanmean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4ee48a46aaa54441b68f05add0610f17","collectionName":"torch.Tensor.nanmean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nanmedian.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"04e738e1190d47d389208772512fb922","collectionName":"torch.Tensor.nanmedian","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nanquantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17a6d8e4d1f54bf9aee7578d2d9fe5eb","collectionName":"torch.Tensor.nanquantile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nansum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b2c597e91ad4ccd9ec26bdf37df5555","collectionName":"torch.Tensor.nansum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nan_to_num.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"10aa9094b2fe4d2da2b9b3855469480d","collectionName":"torch.Tensor.nan_to_num","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nan_to_num_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"99944eff980947c48159de79568b04b0","collectionName":"torch.Tensor.nan_to_num_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.narrow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8a7fd13935064429bf43efa5143742bc","collectionName":"torch.Tensor.narrow","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.narrow_copy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d40cd7845e614196931b2352aafe54a5","collectionName":"torch.Tensor.narrow_copy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ndimension.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fb58b968840c47748419becedc5232fc","collectionName":"torch.Tensor.ndimension","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ne.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ffd6b748e60549c7916fb915c3f789ac","collectionName":"torch.Tensor.ne","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.neg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"891f9c4d897a4d29b4f47193a6109b91","collectionName":"torch.Tensor.neg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.negative.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe968b441897476c8c64b4ee42bce5cd","collectionName":"torch.Tensor.negative","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.negative_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4ef2cb7067334f679e8919ef523f6df3","collectionName":"torch.Tensor.negative_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.neg_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a9a97a9648ad4f0a8f60cefa7074c7f8","collectionName":"torch.Tensor.neg_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nelement.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"41cbabe8a70f4b76beab49342ad0e9aa","collectionName":"torch.Tensor.nelement","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.new_empty.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c721d29654f743b8abd124f767d79da9","collectionName":"torch.Tensor.new_empty","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.new_ones.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6c3b436c06184a739a8281bf3486ed30","collectionName":"torch.Tensor.new_ones","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.new_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"76b8deb69f894a43a09af0dc6799ef61","collectionName":"torch.Tensor.new_tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.new_zeros.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7a33ebec9b8348649aab20f00e08a615","collectionName":"torch.Tensor.new_zeros","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nextafter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8bf1aa7db9a48c3a954ba6b2cfcea87","collectionName":"torch.Tensor.nextafter","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nextafter_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad6a35ecddff45bea03bae3d3f8fd3dd","collectionName":"torch.Tensor.nextafter_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ne_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"211940f7c6554dd5914a0262b78218f4","collectionName":"torch.Tensor.ne_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e172f03d5d304f9c80de5c8073b03292","collectionName":"torch.Tensor.nonzero","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a22c5c0f88ba4a1297b168832c77bf72","collectionName":"torch.Tensor.norm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.normal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c769c35ee3240b88e6e5a7593343bef","collectionName":"torch.Tensor.normal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.not_equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dee4a5bba5f043e1b2f968765adfcf0b","collectionName":"torch.Tensor.not_equal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.not_equal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0a0c245e5353464a8b8d6609096cabcb","collectionName":"torch.Tensor.not_equal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.numel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0aecff5937224143a1698c522ab52bfc","collectionName":"torch.Tensor.numel","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.numpy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2a6792195df743469caa9a225361f461","collectionName":"torch.Tensor.numpy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.outer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ed34578bb0364f74819296717ed147e4","collectionName":"torch.Tensor.outer","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.pinverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a384604c0bf74a77b661af5eefd9016d","collectionName":"torch.Tensor.pinverse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.polygamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d8f31632b9c425e87c952693201e84a","collectionName":"torch.Tensor.polygamma","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.polygamma_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"730e4bd3275249e28c5a409c8ffae22a","collectionName":"torch.Tensor.polygamma_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.positive.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c1a3edf8eb5148c3a15f3de0cfc135bc","collectionName":"torch.Tensor.positive","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.pow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db4daff41d8749cdb55a3e5fb744704e","collectionName":"torch.Tensor.pow","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.pow_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"51c96f16d06e404b87e2f413bdf4a7c7","collectionName":"torch.Tensor.pow_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.prod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5ee788eec06443e86e8994e717bf91b","collectionName":"torch.Tensor.prod","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.qr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9132d3feda8b474e807311b08f59df6c","collectionName":"torch.Tensor.qr","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.quantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3145613a684f412f8b75f519aa614c8e","collectionName":"torch.Tensor.quantile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.rad2deg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7262917bfd424963a9ab3dd9ed1a5530","collectionName":"torch.Tensor.rad2deg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.random_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ed51472916e545098330fd19df4788ac","collectionName":"torch.Tensor.random_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.ravel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a66670ca08ec432f8f80d070e8986d57","collectionName":"torch.Tensor.ravel","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.reciprocal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0673a2a0c69c4f62b40ce07464c272f1","collectionName":"torch.Tensor.reciprocal","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.reciprocal_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9ff088b19984cd1918385edbf96b3e5","collectionName":"torch.Tensor.reciprocal_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.remainder.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3c51c5253d1e4865bd68b9f1eaf973ea","collectionName":"torch.Tensor.remainder","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.remainder_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e643f0219435431aab2e8deab1638c7e","collectionName":"torch.Tensor.remainder_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.renorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b5a9666fce44670863484031d4c221f","collectionName":"torch.Tensor.renorm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.renorm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5c319fe5a12649899e3a4e3a6e1b1898","collectionName":"torch.Tensor.renorm_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.repeat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4aa5abf6249045158ff978800f3e1467","collectionName":"torch.Tensor.repeat","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.repeat_interleave.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e622a42d4e34489b807ac7ef1b907439","collectionName":"torch.Tensor.repeat_interleave","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.reshape.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1c8f4b12f7ad48988a9030de689a243f","collectionName":"torch.Tensor.reshape","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.reshape_as.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"64ddebb0e2094875aeb34b28aca6ed29","collectionName":"torch.Tensor.reshape_as","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.resize_as_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e06a07f8963d491dba914090f4aa6232","collectionName":"torch.Tensor.resize_as_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.resolve_conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a0744f119c884f6198eeaf442ef499a6","collectionName":"torch.Tensor.resolve_conj","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.resolve_neg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc963257751c4730b60b1c16f0d3332a","collectionName":"torch.Tensor.resolve_neg","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.roll.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a3192b6fd77b4b83a80371016fb1406c","collectionName":"torch.Tensor.roll","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.rot90.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c19ef63d95cc4aedbf27d26750b0c11c","collectionName":"torch.Tensor.rot90","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.round.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e8b44ee0340e413d9210e8c0e1ad18fe","collectionName":"torch.Tensor.round","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.round_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b33035505df44171a2efa4560d09a25d","collectionName":"torch.Tensor.round_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.rsqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f8b77b7dbb0c4d1aabe4e232991eef4f","collectionName":"torch.Tensor.rsqrt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.rsqrt_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8d7efb631d5945f186f8001d00051c33","collectionName":"torch.Tensor.rsqrt_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.scatter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"21e6e3d17dee415db3edce47d40e5562","collectionName":"torch.Tensor.scatter","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.scatter_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b75f1ad74fb4356b754fc687f6cef2a","collectionName":"torch.Tensor.scatter_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.scatter_add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cfd3443530fb4a628d1091a40657ca0c","collectionName":"torch.Tensor.scatter_add","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.scatter_add_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b0c57ee90a3646d0a977ff74a4ac6938","collectionName":"torch.Tensor.scatter_add_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d4e30f66d1274cf6b26938dcc8bd036c","collectionName":"torch.Tensor.select","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.set_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3fe8f50e8add4831a870309ec856e6db","collectionName":"torch.Tensor.set_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sgn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c8963dbbad374718b49cb01a9636a8fa","collectionName":"torch.Tensor.sgn","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sgn_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6f17e1a1bd68464097ea1caae85f8061","collectionName":"torch.Tensor.sgn_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.share_memory_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"01075b4b694f493f8649ace4195c43ac","collectionName":"torch.Tensor.share_memory_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.short.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"80f700f415274a01901352c1187935f1","collectionName":"torch.Tensor.short","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3fd7f7ae533f47cc82bbdf28fb8a93f5","collectionName":"torch.Tensor.sigmoid","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sigmoid_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe9aeeb361cc43039c5949d8db89c3a3","collectionName":"torch.Tensor.sigmoid_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a9201a78b51147e0b0ec673574a685f3","collectionName":"torch.Tensor.sign","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.signbit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9112cd1e6d4042f0b61aa89ec2073491","collectionName":"torch.Tensor.signbit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sign_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"80bbfb0f4b3141ddb4ff83bda702d08f","collectionName":"torch.Tensor.sign_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"626900386c2d42d79ca4bb74e5a1765f","collectionName":"torch.Tensor.sin","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sinc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"25b108b0de4548c8aa6d2c25b0c73286","collectionName":"torch.Tensor.sinc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sinc_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b3458b456eda4b6ca5676d2064b7f4bf","collectionName":"torch.Tensor.sinc_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f3e5d3dd40734527828c86e6ab776498","collectionName":"torch.Tensor.sinh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sinh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f7fedec2fa70403489b6a78a6652957f","collectionName":"torch.Tensor.sinh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sin_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"14e6b524a43948cb9b9b10beeadb096a","collectionName":"torch.Tensor.sin_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.size.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e1dd73625514055a4e0750613bd0ae5","collectionName":"torch.Tensor.size","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.slogdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fd218aa15af34ab582f2f171db5ae04a","collectionName":"torch.Tensor.slogdet","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.smm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"31f36a217ffc4b059e87ea69cdb8e48b","collectionName":"torch.Tensor.smm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d25f85aca6b2402f8478c5093b2c9cdf","collectionName":"torch.Tensor.solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2affae6dde9b4da1afd421dbf8888b03","collectionName":"torch.Tensor.sort","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b830a55a78844774bb7e6a912b6cd04a","collectionName":"torch.Tensor.split","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"056ffa7d54614cbd9d52348d66a3c2e6","collectionName":"torch.Tensor.sqrt","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sqrt_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9c933c5277644ba1b09f55999b890778","collectionName":"torch.Tensor.sqrt_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.square.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1b2b19cc6dc0407991919595de622005","collectionName":"torch.Tensor.square","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.square_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"08cb938bc86f4665bf9174cbfc760380","collectionName":"torch.Tensor.square_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.squeeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96818e8a2405400587001e697fae5a6c","collectionName":"torch.Tensor.squeeze","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.squeeze_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b0ee7c7318bd4ccd89045675bad8a7d1","collectionName":"torch.Tensor.squeeze_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sspaddmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95743e0524eb496f83e268353355eb38","collectionName":"torch.Tensor.sspaddmm","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.std.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6681903746224e89b365483445d9a2fa","collectionName":"torch.Tensor.std","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.storage.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7bd5f211632c4c45bf421b3757d33542","collectionName":"torch.Tensor.storage","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.storage_offset.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c657209056f4d3fb68bb3d94cd26f6f","collectionName":"torch.Tensor.storage_offset","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.storage_type.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c7b9b09b6f6145afb4bdb1b3b6273d5e","collectionName":"torch.Tensor.storage_type","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.stride.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"02ca0661d0c94aab834ba36fcecfb5dd","collectionName":"torch.Tensor.stride","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sub.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d24655764d194d8888792e20620792aa","collectionName":"torch.Tensor.sub","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.subtract.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"75c804185cd84558869be045a23fd50c","collectionName":"torch.Tensor.subtract","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.subtract_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc4ad9b0ba384e3da40bedc78072eea2","collectionName":"torch.Tensor.subtract_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sub_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b304d7e955f47f1a1963b83bc509b0d","collectionName":"torch.Tensor.sub_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"021d639333f34059bb140f4cef450fac","collectionName":"torch.Tensor.sum","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.sum_to_size.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"da6ab3d895924495aba29d0b048365b1","collectionName":"torch.Tensor.sum_to_size","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.svd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6501598b90b44ed798f254be3154d20c","collectionName":"torch.Tensor.svd","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.swapaxes.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b79df8f91045463a80b608a2f5d40aea","collectionName":"torch.Tensor.swapaxes","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.swapdims.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2643580cbfeb47d297a863f9cbc70ac5","collectionName":"torch.Tensor.swapdims","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.symeig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e58816eae27435ebf10b86d36620914","collectionName":"torch.Tensor.symeig","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.t.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9a79b687715d48a498f0a1801f09fcb4","collectionName":"torch.Tensor.t","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.take.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2d590fdb87884239bebf36da747c120b","collectionName":"torch.Tensor.take","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33018874343d4a9fb2237e75f395a315","collectionName":"torch.Tensor.tan","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad7b5c2da6e147f4840ab93186fb8898","collectionName":"torch.Tensor.tanh","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tanh_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f6ee0648892c46ce8a5ec6a4f904748d","collectionName":"torch.Tensor.tanh_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tan_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ce2c9d300b4f40b19d6da324e91c3aff","collectionName":"torch.Tensor.tan_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tensor_split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5adbe79ac1b4a589505318dd0851f49","collectionName":"torch.Tensor.tensor_split","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b6b319a983a842fcb6ca67b323c5e089","collectionName":"torch.Tensor.tile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.to.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33561279af3a486e8817bfca79f83a01","collectionName":"torch.Tensor.to","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tolist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b89747d5c16f4ee38ce1eed910a5bf9e","collectionName":"torch.Tensor.tolist","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.topk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a928acbca63b43058036add61895a51f","collectionName":"torch.Tensor.topk","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.to_sparse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b78a975f3b0f4e779fd8734e132e1050","collectionName":"torch.Tensor.to_sparse","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.trace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"900cacdce09e4812859bc2a05979ee37","collectionName":"torch.Tensor.trace","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.transpose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"50257e28651e4a128694165253b04db8","collectionName":"torch.Tensor.transpose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.transpose_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9efdc80f6ee445c8818c0d5f6d1608e","collectionName":"torch.Tensor.transpose_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.triangular_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f6bb86935aa74fef91ffe374da21251a","collectionName":"torch.Tensor.triangular_solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tril.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2677b6cb7bab4e96b5a253869917bd1d","collectionName":"torch.Tensor.tril","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.tril_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4512961c2bf04ec688e469588bc30aaf","collectionName":"torch.Tensor.tril_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.triu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d1d9f227379b4dd98a2b2f42e21e5ef6","collectionName":"torch.Tensor.triu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.triu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a374164f4ad4039b15982af80dc0bcb","collectionName":"torch.Tensor.triu_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.true_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b36e0981ae9414fa4cee51527a374f1","collectionName":"torch.Tensor.true_divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.true_divide_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ed4657e1db54893b31f0d9b97b55abb","collectionName":"torch.Tensor.true_divide_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.trunc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"84f112114e924d4e84db998acccea70d","collectionName":"torch.Tensor.trunc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.trunc_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"368d5da5fb404d21be4acfd209d23c0b","collectionName":"torch.Tensor.trunc_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.type_as.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f090f3fc8ca74f8980b5c107a980b039","collectionName":"torch.Tensor.type_as","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.t_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e7a91c6df0444eb586e6d283adf2aba2","collectionName":"torch.Tensor.t_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unbind.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f485c8ae071b452bb9ebdf5d41f31d24","collectionName":"torch.Tensor.unbind","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unfold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"02cd412c6e4a42049c2d8a8885843796","collectionName":"torch.Tensor.unfold","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.uniform_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4feae4004a7e478b886918bf693c9a79","collectionName":"torch.Tensor.uniform_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unique.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e80e1b70e104b7abafa3397729a1c34","collectionName":"torch.Tensor.unique","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unique_consecutive.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"242195b1f842433fbafe291b967dc890","collectionName":"torch.Tensor.unique_consecutive","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unsqueeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb06d11dd42c4df2bd1f409363b24f1b","collectionName":"torch.Tensor.unsqueeze","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.unsqueeze_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a057682f655433e813e284fbd2610a6","collectionName":"torch.Tensor.unsqueeze_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.var.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b2c692216e643e995751174e2adfe9a","collectionName":"torch.Tensor.var","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.vdot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fb7c40135d3441e9b7b37fdc8f4308cb","collectionName":"torch.Tensor.vdot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.view.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4dc1523a138a45efa97dada0a1a2b1ec","collectionName":"torch.Tensor.view","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.view_as.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"97f6198d47f7492a860ad17d1aeae209","collectionName":"torch.Tensor.view_as","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.vsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"afff20b2a04f4249a747d6aa0157ecdd","collectionName":"torch.Tensor.vsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.where.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b2cf9e6d1a5c4d7ab3b0b4b772ea7a6e","collectionName":"torch.Tensor.where","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.xlogy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"350ea582f8cf400f88facd8bd1724748","collectionName":"torch.Tensor.xlogy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.xlogy_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ca635eab722435aa4f3e8190441a78a","collectionName":"torch.Tensor.xlogy_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.Tensor.zero_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6a769714949a48ea8bb351a29e70e073","collectionName":"torch.Tensor.zero_","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tensordot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e992db6df9945e2b857627ef29a905c","collectionName":"torch.tensordot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tensor_split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"510f7549b96f4c1591cb23b728c1e4fa","collectionName":"torch.tensor_split","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.testing.make_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ae07d8b89734c14b269fa6b522d7667","collectionName":"torch.testing.make_tensor","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3ee70b1567cf452eb6a6003483985769","collectionName":"torch.tile","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.topk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d92267067ea3477c8284a26a802e41b3","collectionName":"torch.topk","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.trace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"59e668e5c01241a997373b6dcd5c78ac","collectionName":"torch.trace","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.transpose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d94b0f6b1978440f822c7f5b15011627","collectionName":"torch.transpose","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.trapz.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"66d4075ee5d946be9b2ae2fe78b78a1b","collectionName":"torch.trapz","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.triangular_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"031f51f765684336a31c3fb61a673b91","collectionName":"torch.triangular_solve","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tril.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"15deb219b76040428d2257bc31692e1d","collectionName":"torch.tril","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.tril_indices.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e851489e369649d498d357f8794ec499","collectionName":"torch.tril_indices","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.triu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d7384fe849bb4dfe832ce1172e15f0f7","collectionName":"torch.triu","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.triu_indices.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2cae8ee359b74e6b9ab9bcc3eb5bc4c6","collectionName":"torch.triu_indices","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.true_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2be3c8d0caec494195e582fda79cf059","collectionName":"torch.true_divide","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.trunc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d9c60f424e9d49f981ba101584cfea48","collectionName":"torch.trunc","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.unbind.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d0204058a8534fc48bc7563bf5517e31","collectionName":"torch.unbind","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.unique.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33f67a7a51e645e8b46112bda4181f71","collectionName":"torch.unique","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.unique_consecutive.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e318b560d9234c1392f90dbe005c59db","collectionName":"torch.unique_consecutive","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.unsqueeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eb65d070a1924648b0500395f205e77f","collectionName":"torch.unsqueeze","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.use_deterministic_algorithms.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bd4d09a75c604c27b61d6b8c5fb0e69c","collectionName":"torch.use_deterministic_algorithms","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.utils.dlpack.from_dlpack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"14b0223f12fe475ebc1887d487f29bde","collectionName":"torch.utils.dlpack.from_dlpack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.utils.dlpack.to_dlpack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bc7fcdf4ac7944c8ac5c940645d93227","collectionName":"torch.utils.dlpack.to_dlpack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.vander.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"03b1251c6ed84441b46530cf84e712db","collectionName":"torch.vander","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.var.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"104a276fc3054815b44ec822441c3af8","collectionName":"torch.var","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.var_mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ec6a744d89c84eb7a2426e5bcaa39ef6","collectionName":"torch.var_mean","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.vdot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4bb0c3ce42fc4486a952b80a39641eb0","collectionName":"torch.vdot","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.view_as_complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c6e4a0a7f5fa4f4781e97b9d9a1f3f5b","collectionName":"torch.view_as_complex","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.view_as_real.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9fde0fc66c0a4ddd9c0b89b04505c7ac","collectionName":"torch.view_as_real","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.vsplit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"502cb971e1c34f5ebb6796ec982fca63","collectionName":"torch.vsplit","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.vstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1674624478ed41a18fc549d9dc7331e9","collectionName":"torch.vstack","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.where.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d2cc93f786904b0fa95f3fc0b2ca1252","collectionName":"torch.where","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.xlogy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"edd998f54a6b43fa935a974b3e45914d","collectionName":"torch.xlogy","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.zeros.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6704893829b84dddbef7480fd31c7c1d","collectionName":"torch.zeros","type":"collection"}

================================================================================
FILE: dump\deeprel-torch\torch.zeros_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b0e4a187d654c679efd495709bea0e8","collectionName":"torch.zeros_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\argVS.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6fbf36f37cd24f77a4f5df6a1a59733a","collectionName":"argVS","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\signature.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5195bab6dd44d218ebdf72f84e1890c","collectionName":"signature","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\similarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a89ef85e0f074b07a84239734ad29f2f","collectionName":"similarity","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.abs.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bf258d37163a43d29be5b50d8b52d3e4","collectionName":"torch.abs","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.acos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5559f238f3ed40248433c1b88eb6d62f","collectionName":"torch.acos","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.acosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a562974a33144dcf834e56c8f0fa9103","collectionName":"torch.acosh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ef59f159b6c842229fdbb62ac195864f","collectionName":"torch.add","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9aabce4990cc4592b644451fcf7da364","collectionName":"torch.addbmm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addcdiv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"34e08ffdf27a4758a3acba7c9d1ba25d","collectionName":"torch.addcdiv","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addcmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b686fa2cfe54fdd8c95cd336dceac0b","collectionName":"torch.addcmul","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56f24b57d1f046c7992d1080c2921aaf","collectionName":"torch.addmm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addmv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"54df2f85aef54dc6a91de75955085177","collectionName":"torch.addmv","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.addr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bab17d0cba2b4fc29b585d915d4f84e5","collectionName":"torch.addr","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.allclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c6c3690a226f4d43a7064761e7d2b763","collectionName":"torch.allclose","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.amax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d7f920970a1f42f1998d055359a2786e","collectionName":"torch.amax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.amin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a58ee5ecc0ab46ad94d9705df05611ba","collectionName":"torch.amin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.angle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"349e80bfc6a8491b80755563ea270ec4","collectionName":"torch.angle","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.arange.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"38bc512855fb47afa8a277c0914788ae","collectionName":"torch.arange","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.are_deterministic_algorithms_enabled.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"631251a018d24e2fb84dfe5319d6f897","collectionName":"torch.are_deterministic_algorithms_enabled","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.argmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9f84e27e429a4f5fa7f1b3798489a3dc","collectionName":"torch.argmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.argmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7a4c40acaf014f56ab92f37973d41eff","collectionName":"torch.argmin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.argsort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c46245724f854c14a09ba8e16b9fbf10","collectionName":"torch.argsort","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.asin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e7fcd58165e4ef68c28b49abef39ad6","collectionName":"torch.asin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.asinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"82ef50839c8a44aba3e5edd5e1eb5a3b","collectionName":"torch.asinh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.as_strided.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ced00f90c78448a5a4ee52a7b7dc4ad7","collectionName":"torch.as_strided","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.as_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"98e514b9c74944b28221566d65dd8686","collectionName":"torch.as_tensor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"be26135903674a1da48ba40107aabbed","collectionName":"torch.atan","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atan2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"84d593e11795416eb4edec56caf0df9f","collectionName":"torch.atan2","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3647253546ff4309bcb244186860f2f8","collectionName":"torch.atanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atleast_1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"94bf40a0dc0f4bc3beea648b29fc65d2","collectionName":"torch.atleast_1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atleast_2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4f29d15e4ca549d196e0dffa80fb4063","collectionName":"torch.atleast_2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.atleast_3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"81549ecee1f0423f9ff01af1a2dcec88","collectionName":"torch.atleast_3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.baddbmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"097d6511af344cb7b0eda72729392ceb","collectionName":"torch.baddbmm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bernoulli.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"94f4ea75de0e420db4f9ba2a3dc61189","collectionName":"torch.bernoulli","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bincount.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0298443da37348459bfda7532d445bec","collectionName":"torch.bincount","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bitwise_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e413b8bb1518455ca117f2a9abb735fa","collectionName":"torch.bitwise_and","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bitwise_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1c54fcc135d7442cb81b83b99d4cae2b","collectionName":"torch.bitwise_not","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bitwise_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bc04d40a660d4492a27f34ad5ad05733","collectionName":"torch.bitwise_or","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bitwise_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4e9dbdfdef37401c877cc6ec10315bed","collectionName":"torch.bitwise_xor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.block_diag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"27419c6ef63c4f47ad34538c3ca4db65","collectionName":"torch.block_diag","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bmm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ceb787d5411c44cbbdb2345566e229b5","collectionName":"torch.bmm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.broadcast_shapes.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8bb1c2adafa043dd88cc70de92f310c7","collectionName":"torch.broadcast_shapes","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.broadcast_tensors.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a9a298bb5474b6ea944e0bd669a3b26","collectionName":"torch.broadcast_tensors","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.broadcast_to.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"61702049a4fd4979ac07f5d1dc3d6646","collectionName":"torch.broadcast_to","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.bucketize.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cba7007da039461c97ae8aabb8b658fb","collectionName":"torch.bucketize","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.can_cast.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d14286067fff4904ba90a9266635ca8a","collectionName":"torch.can_cast","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cartesian_prod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4fee54ee8efd4bb7b8d5b122944ee88f","collectionName":"torch.cartesian_prod","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"012945e8fd414055914979caccdf53b9","collectionName":"torch.cat","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cdist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4372fe55702647f9b6311a6d087b0775","collectionName":"torch.cdist","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ceil.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2542951995704bf494f3e0a910f5e4e7","collectionName":"torch.ceil","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.chain_matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"835306aa026e483d8ffb5652f6a5c330","collectionName":"torch.chain_matmul","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cholesky.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"40e8a45b886f4c9b89f81e1e14a33f4d","collectionName":"torch.cholesky","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cholesky_inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f735c33e043445e983231c6a2394f845","collectionName":"torch.cholesky_inverse","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cholesky_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"045cc31bf6974d43a868ea7999521a5d","collectionName":"torch.cholesky_solve","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.chunk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dc3498cefe8b4a1d97f1216bc4642c6c","collectionName":"torch.chunk","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.clamp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c45c78abe3aa42698f7a16b34db727a5","collectionName":"torch.clamp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.clone.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c48c238c47d248c19d1b870c18995c5a","collectionName":"torch.clone","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.combinations.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"864b34ea4ba4419eb848019e24d502f9","collectionName":"torch.combinations","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7d974dde4ae046428798a9d158588978","collectionName":"torch.complex","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.conj.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5b233028d49e4c74b654577fa15ff422","collectionName":"torch.conj","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.copysign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"697d0f884f6549c9b49c267bb7978b0c","collectionName":"torch.copysign","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cos.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5d21e8b6449d4df7bd7f577251bad319","collectionName":"torch.cos","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cosh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"02a597c8c9a848a087288833a436ea60","collectionName":"torch.cosh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.count_nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"04b5199bc0274a56afc463c694fad36f","collectionName":"torch.count_nonzero","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cross.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1eb5b31672924516a2bbad7acac3ac74","collectionName":"torch.cross","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cummax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3059fe387672496592562e1da75f8cd7","collectionName":"torch.cummax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cummin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dbdd05886b0448a0896b2d5dacd698f6","collectionName":"torch.cummin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cumprod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ec7c2d1ce5d84fe986e67f233c845719","collectionName":"torch.cumprod","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.cumsum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e7946724c49b48d993c30f78b98ad5a0","collectionName":"torch.cumsum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.deg2rad.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"12dafc7377dd4f11acbb28bfe1593af5","collectionName":"torch.deg2rad","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.det.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"546b0327edfc4bcda4c512ed64f419ae","collectionName":"torch.det","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.diag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"49747e4c49c041b5a3d7632e75f2dc68","collectionName":"torch.diag","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.diagflat.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d997fc177fa440029e67e05a42a31c29","collectionName":"torch.diagflat","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.diagonal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b8c75744731445389f06b5159880a46","collectionName":"torch.diagonal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.diag_embed.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f01552aaccc643218c17a50f550253e3","collectionName":"torch.diag_embed","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.digamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d008cfecad142039c890984b9f37145","collectionName":"torch.digamma","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.dist.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6b8b678419424457af75c4f5858f93bb","collectionName":"torch.dist","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.div.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c4e8ec28b0244d6a9613ecd121f3945e","collectionName":"torch.div","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.dot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6396d083e12c4d2bb56f15d8e1edda01","collectionName":"torch.dot","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.dstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4512d6115a174bfbb9c35d90a8fafbba","collectionName":"torch.dstack","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.eig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"14df40a3dbdd4188a9edb41fd125fc75","collectionName":"torch.eig","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.einsum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a132d7611f954c70a47f8e9266f7241b","collectionName":"torch.einsum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.empty.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba1ca66760d64e458cade3e40d223a9a","collectionName":"torch.empty","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.empty_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cb110315df984c2b90f056b0de398519","collectionName":"torch.empty_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.empty_strided.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"88f3c59398ff4775a5fd6703e8e8176f","collectionName":"torch.empty_strided","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.eq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8200b417397d4c2f8b4945c0b621b3d8","collectionName":"torch.eq","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.equal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6938ee1b06d34294b497589b0266891b","collectionName":"torch.equal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.erf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"51371870efbf404ba9dbb14b56ead86e","collectionName":"torch.erf","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.erfc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1e177408f7b5495fa6d392f17064749a","collectionName":"torch.erfc","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.erfinv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b902a794fbd84216ac0467379aa54617","collectionName":"torch.erfinv","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"66fd36a73f504f5b83e820355ccf6395","collectionName":"torch.exp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.exp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"39fd118b33804dc0a04b94c0c478620e","collectionName":"torch.exp2","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.expm1.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0409983f98ff4efba794deaa2f59809b","collectionName":"torch.expm1","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.eye.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1966bae820304a3291047502ce3d4689","collectionName":"torch.eye","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.flatten.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9bb38bf24ff44c059000f19bc3ad92d6","collectionName":"torch.flatten","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.flip.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2c7803014e2447929e554e89ebab2f76","collectionName":"torch.flip","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.fliplr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b4e3a24860444bcaa3ac480f1b4e336e","collectionName":"torch.fliplr","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.flipud.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"13482b1ce5d64d78a271e6c3a8db4721","collectionName":"torch.flipud","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.float_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"39e31a8445dd4244bdee3b3804263a8b","collectionName":"torch.float_power","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.floor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9d9f90702d6f44358a35bd3545d17380","collectionName":"torch.floor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.floor_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5ea3473acc048e7ad04bc0d46ddab52","collectionName":"torch.floor_divide","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.fmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6af26bb6cb8b4d6798c57eedf784d8af","collectionName":"torch.fmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.fmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ee294bfc87024598b08b540340884742","collectionName":"torch.fmin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.fmod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e4c26b2caae94755b3ca29d98123501c","collectionName":"torch.fmod","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.frac.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0e0d914f320b4078a757e48868406007","collectionName":"torch.frac","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.full.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3e86b9603989442fab70605155661a07","collectionName":"torch.full","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.full_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"610087cec6e04813a513ede2e470c0f2","collectionName":"torch.full_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.gather.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"beecfd9b902646c8a5b5f204813f3f5d","collectionName":"torch.gather","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.gcd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d1af1eb319b740868a0bfebc31db5ad1","collectionName":"torch.gcd","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ge.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9d2c3879bb742559ce20ef9ceb34426","collectionName":"torch.ge","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ger.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a7c84edbb32a4c6bb468533808b21b19","collectionName":"torch.ger","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.get_rng_state.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3bc1e5d6fd114130b870f9a55bd1bc0d","collectionName":"torch.get_rng_state","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.gt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cab7855cd3244f7e9abc247e14ef1709","collectionName":"torch.gt","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.heaviside.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dbf415bb33c947d9aff2278ff360ba64","collectionName":"torch.heaviside","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.histc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5e52ae5bd0ac460c8d2217d1768e6575","collectionName":"torch.histc","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.hstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57c5f17c601d4f4abf11a52e4ff9fac9","collectionName":"torch.hstack","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.hypot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7d46974fd6b547fb93139f5a4ec08989","collectionName":"torch.hypot","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.i0.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7ecf4bc093354f218a82427f45d4f4e8","collectionName":"torch.i0","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.igamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"084b04298f554ac99b8bc73b0157cb9f","collectionName":"torch.igamma","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.imag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f7303d1ba9a34870bef7ea2d6efbc5ef","collectionName":"torch.imag","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.index_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3fe96a13bdbd4db385819d4da60341f9","collectionName":"torch.index_select","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.inner.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"488c5be9cbe5443fbba7cb1c490637ed","collectionName":"torch.inner","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.inverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f69b749af6e4ed4a67202dcfe3cd481","collectionName":"torch.inverse","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isclose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1d3e52922fac44ab9aab74bde1e17f7a","collectionName":"torch.isclose","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isfinite.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2d6df9f5c8a94f17a8f2f647ee3cbeb7","collectionName":"torch.isfinite","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d4ec9e168ba4f0e976b8e9e5e033bdb","collectionName":"torch.isinf","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isnan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"500feb1c3c844263bc6e17b1e6e6544a","collectionName":"torch.isnan","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isneginf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0707f35bb9f748228994c16fb77687f2","collectionName":"torch.isneginf","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isposinf.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a859bb1374ed44cbbb609e11ead9ea22","collectionName":"torch.isposinf","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.isreal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1bb06be6362e4225b3a0af0f2d6e3bfc","collectionName":"torch.isreal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.is_nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e43e5326830d4c62a44ddd882fba0cd8","collectionName":"torch.is_nonzero","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.is_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e13e901bac014baa90624f2aec09ee55","collectionName":"torch.is_tensor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.kron.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3509633f5134487bab69fe1d63db59ee","collectionName":"torch.kron","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.kthvalue.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"83406fad527342439b80ccede90efd0c","collectionName":"torch.kthvalue","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lcm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b5fb53146ce8448eb80112f9044c7030","collectionName":"torch.lcm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.le.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3e464004b9cf4aad898d9021d8ef986b","collectionName":"torch.le","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lerp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9d4e6f6dbba54c1c8c01829b3834ae13","collectionName":"torch.lerp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"972a55ffae4d44e38fd26c4ba6eafd2a","collectionName":"torch.lgamma","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.linspace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0c1dbcae20b74438af81322f57fdccab","collectionName":"torch.linspace","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.log.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2feacffbfabc420d942de39414e45fab","collectionName":"torch.log","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.log10.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2064030b1f264f03a74b5cacf963434c","collectionName":"torch.log10","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.log1p.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e7d219c9c9841d993a306e684314850","collectionName":"torch.log1p","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.log2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"035ce01943764a58af3e6e4d53ec4c9b","collectionName":"torch.log2","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logaddexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"509c14810d1044f2bd7575b5b0c3b114","collectionName":"torch.logaddexp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logaddexp2.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc5b566458db4c29b0d671d13180bdaa","collectionName":"torch.logaddexp2","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logcumsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"924fc7d55d39485ca5c3543185370007","collectionName":"torch.logcumsumexp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ee30c6adf1d47268670ad55adeeee6b","collectionName":"torch.logdet","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logical_and.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d1658bbcd9ae4521b4f5963919520cba","collectionName":"torch.logical_and","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logical_not.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"68504765ba4845229a8f7da6169f5750","collectionName":"torch.logical_not","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logical_or.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"34966cc2837d422b885b1c65744024b4","collectionName":"torch.logical_or","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logical_xor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1cc71984fcc04e15aec3cdff83854832","collectionName":"torch.logical_xor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d88d06c813ad467492496ca48e1afab9","collectionName":"torch.logit","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logspace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b1c68575c57a469580431e0f24f33a9b","collectionName":"torch.logspace","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.logsumexp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d4e07848a3847aaad4b582374c25ab9","collectionName":"torch.logsumexp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lstsq.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"50ad45d235704071a5a901280c24abc4","collectionName":"torch.lstsq","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3e753c9d46774770a3eb456f056c86e2","collectionName":"torch.lt","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ab75d361151d4074984ee0f706ce1378","collectionName":"torch.lu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.lu_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ddef19c6eca447a5bdd69a1d6c0af3ff","collectionName":"torch.lu_solve","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.masked_select.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2731496c5c614bf7b2993d69f6bd6bdd","collectionName":"torch.masked_select","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.matmul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ffd0d6c2bbf4b039fc1c1dee0693fbf","collectionName":"torch.matmul","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.matrix_exp.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5250deff4f3e4af69f96a25ac8a29743","collectionName":"torch.matrix_exp","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.matrix_power.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2a18ea38d4f54e7294fc5057232aea65","collectionName":"torch.matrix_power","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.matrix_rank.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3998b8266d814d7194c2af883a241bd2","collectionName":"torch.matrix_rank","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.max.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b22007fb73b44718a77ebe82dcc1f98a","collectionName":"torch.max","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.maximum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b4f8fbfbe50c43c4adec9312f7f7eae5","collectionName":"torch.maximum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2583d244259745e3ba03aa95217a7812","collectionName":"torch.mean","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.median.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57447132b56c457f9ade4057e850ed6d","collectionName":"torch.median","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.min.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a769757595514559afbec71f692346d1","collectionName":"torch.min","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.minimum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"95c104f0059642ee94e0dd901dd981cf","collectionName":"torch.minimum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e64dac3894544cc88224e8f8f6297e35","collectionName":"torch.mm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mode.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b06cdf7095ac4e5b87cdea2e9c69fb04","collectionName":"torch.mode","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.movedim.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3dc783079dbc47beb8d1d44ea8670256","collectionName":"torch.movedim","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.msort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1e07ebc11095465cae8b33eb1c299f60","collectionName":"torch.msort","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mul.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0b10022a9ba647648065ebeff69ae001","collectionName":"torch.mul","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.multinomial.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4ce492e29ca4438fb032d6b7af4a4999","collectionName":"torch.multinomial","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mv.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0142f49753ab4f2597e98094eabe605f","collectionName":"torch.mv","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.mvlgamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b5d34793e1af4f179bffd0cb95bd7ddd","collectionName":"torch.mvlgamma","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nanmedian.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"40aba82c034a42be9167f76c1ac45b8f","collectionName":"torch.nanmedian","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nanquantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"889a59c537a2447c8af2f2a6c813bbec","collectionName":"torch.nanquantile","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nansum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ea3be071a9344dbf991411126269a676","collectionName":"torch.nansum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.narrow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"27da9738e86744208858e5b57957e0d0","collectionName":"torch.narrow","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ne.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bedc6234d7a54103b16baae70c6fe3d8","collectionName":"torch.ne","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.neg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e39f53905fe7413e805548b0fbacf5ce","collectionName":"torch.neg","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nextafter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d0ef131cb21f4c9cbcb154ae9407ddcd","collectionName":"torch.nextafter","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveAvgPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3eeba94697e543458e1de5f5e1ae9173","collectionName":"torch.nn.AdaptiveAvgPool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveAvgPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1ce9000de16349ac9834f0aff121894a","collectionName":"torch.nn.AdaptiveAvgPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveAvgPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3f2b86d12c5643ab80be8f8ce2406288","collectionName":"torch.nn.AdaptiveAvgPool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveMaxPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e8ad86dcc18d4a148f737290ab545628","collectionName":"torch.nn.AdaptiveMaxPool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveMaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"88538c301a1740cf8389797908a97803","collectionName":"torch.nn.AdaptiveMaxPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AdaptiveMaxPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dcc55dc9915d4752a87f121fa66a6af8","collectionName":"torch.nn.AdaptiveMaxPool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AlphaDropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5e67192d3a964aba9c6ad1887f5bc8a3","collectionName":"torch.nn.AlphaDropout","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AvgPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"80b2819f4a0a4e388ed76c1f901f40ee","collectionName":"torch.nn.AvgPool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AvgPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f99f7c435b914f43a7ada78283925766","collectionName":"torch.nn.AvgPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.AvgPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7a15b3a6657a4679ae03b9f66b403ac4","collectionName":"torch.nn.AvgPool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.BatchNorm1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"82c289789ca34af6a9ef349a3eeab71b","collectionName":"torch.nn.BatchNorm1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.BatchNorm2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dd7174443c9a4f5aaf6726443df4d90c","collectionName":"torch.nn.BatchNorm2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.BatchNorm3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"534132cbad4e492a934e3bad50b121f1","collectionName":"torch.nn.BatchNorm3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.BCELoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"99712aed286c48b0b6b3d213778d39c8","collectionName":"torch.nn.BCELoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.BCEWithLogitsLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3e010c7f821a4a538b12900bc860eaf0","collectionName":"torch.nn.BCEWithLogitsLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Bilinear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"564d1c3aa9c94fad8e63370f323e60cc","collectionName":"torch.nn.Bilinear","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.CELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cbe94959a1ab4c72a4992bc3193f827b","collectionName":"torch.nn.CELU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ConstantPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dfd15bef6a6d442194eeea68d7bdeb4c","collectionName":"torch.nn.ConstantPad1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ConstantPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9716cc26aed74810a76bf7b1cd6ce2d3","collectionName":"torch.nn.ConstantPad2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ConstantPad3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33b3875c4dd446fda0529a218e10dc76","collectionName":"torch.nn.ConstantPad3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Conv1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b37ab9c73edb421d9ad93a03f5a0a947","collectionName":"torch.nn.Conv1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Conv2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5b410634a5048679398a7873f7172bb","collectionName":"torch.nn.Conv2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Conv3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"074ecbafee5d4495a751af187cfa08da","collectionName":"torch.nn.Conv3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ConvTranspose2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9adc8702ac4d4e46a4a265dbb13389b5","collectionName":"torch.nn.ConvTranspose2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ConvTranspose3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36e0a19c37e1493ea052275498256df4","collectionName":"torch.nn.ConvTranspose3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.CosineSimilarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"43a48581655546a9879b7baee8d349c5","collectionName":"torch.nn.CosineSimilarity","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.CrossEntropyLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d73d193f9c8465a89357628ebefb86a","collectionName":"torch.nn.CrossEntropyLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.CTCLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eaa1d50a22124c9796c86e333c412020","collectionName":"torch.nn.CTCLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6f6e0c0764e245b0a067ec2bddb6f121","collectionName":"torch.nn.Dropout","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Dropout2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0d96b038f361442aa0cf934b34986989","collectionName":"torch.nn.Dropout2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Dropout3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d0c6f5a86e047eebde5402ef045edd5","collectionName":"torch.nn.Dropout3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"03c3a608ccd640459329df5c0af79ec6","collectionName":"torch.nn.ELU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Embedding.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"deb02d1ad32f4570abbdba8ebbc65274","collectionName":"torch.nn.Embedding","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Flatten.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7fc476c0fa494e0dbb5516e4a2e5650a","collectionName":"torch.nn.Flatten","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Fold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8eda5425358042b3a2f289f8e967a5c0","collectionName":"torch.nn.Fold","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.FractionalMaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b5e04d589844b95a4f96ef4a7a7ac77","collectionName":"torch.nn.FractionalMaxPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_avg_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"16ab18a5812341729c38fb607bd0fb66","collectionName":"torch.nn.functional.adaptive_avg_pool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_avg_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2e50790e5ddd46cebcd4fc6481d48337","collectionName":"torch.nn.functional.adaptive_avg_pool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_avg_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dbda1f86985f4adb8525fd726d234332","collectionName":"torch.nn.functional.adaptive_avg_pool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_max_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"01d39e29efd942cf8cc1e90c35ed75e0","collectionName":"torch.nn.functional.adaptive_max_pool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_max_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b64e651279594dd48fadf7da81a271d4","collectionName":"torch.nn.functional.adaptive_max_pool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.adaptive_max_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56ba7e9585c547a79437c3d6909d41e4","collectionName":"torch.nn.functional.adaptive_max_pool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.alpha_dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1893d9900cb64fb5bb00266ee2d42f8c","collectionName":"torch.nn.functional.alpha_dropout","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.avg_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36b9051707d84bb8966c498ea689f997","collectionName":"torch.nn.functional.avg_pool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.avg_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0af7e8a78a454866b48b3c07591ac3ad","collectionName":"torch.nn.functional.avg_pool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.avg_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"10d099443b3d43778cd65ab81edea940","collectionName":"torch.nn.functional.avg_pool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.batch_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"550952a1347b4bd4acd01b4697256318","collectionName":"torch.nn.functional.batch_norm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.bilinear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4951d02fd85c4e5d9355fc77009b8be4","collectionName":"torch.nn.functional.bilinear","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.binary_cross_entropy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cb42941edc45477584dc0246fd8a2913","collectionName":"torch.nn.functional.binary_cross_entropy","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.binary_cross_entropy_with_logits.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"db40bfa9e0f847568217c15cd335468d","collectionName":"torch.nn.functional.binary_cross_entropy_with_logits","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.celu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67e95a7ad2fe42598613476b1e396849","collectionName":"torch.nn.functional.celu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.conv1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b124ef40f2f43ff94733cfc7412d801","collectionName":"torch.nn.functional.conv1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.conv2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0205ef93bb734b938223c84335d3813b","collectionName":"torch.nn.functional.conv2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.conv3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a50c6753e3b471ba6f06bf9af76c9c7","collectionName":"torch.nn.functional.conv3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.conv_transpose2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"93f8cadc7e6d43f292d4ce7bab654d1a","collectionName":"torch.nn.functional.conv_transpose2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.conv_transpose3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3969f00fd9174dbcb1481a16ef03cf90","collectionName":"torch.nn.functional.conv_transpose3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.cosine_similarity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7e25a3f6d33245b59b8dd23e9476df8f","collectionName":"torch.nn.functional.cosine_similarity","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.cross_entropy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3964aac37aa7497f99f20650696d16e4","collectionName":"torch.nn.functional.cross_entropy","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.ctc_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"141805c1496d45d6a15c32946370b09c","collectionName":"torch.nn.functional.ctc_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.dropout.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"94cba681e7684c559c10af57e67cb367","collectionName":"torch.nn.functional.dropout","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.dropout2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"21b9ae9dc6d24c4a82eecd6af9544ad7","collectionName":"torch.nn.functional.dropout2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.dropout3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f2679a28003f419a898705a289a62f91","collectionName":"torch.nn.functional.dropout3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.elu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a076922b14214f5686ec3ca22cf35832","collectionName":"torch.nn.functional.elu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.elu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b1c4462c318f40f19473d7bd934c132d","collectionName":"torch.nn.functional.elu_","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.embedding.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a7499e88cdab41d9a683e3aa5961758b","collectionName":"torch.nn.functional.embedding","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.embedding_bag.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"52c1a7a18b4d4942a9fb07ce5948a968","collectionName":"torch.nn.functional.embedding_bag","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.fold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fcd6c66a109148c1a8d920443bfde843","collectionName":"torch.nn.functional.fold","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.gelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67d7bcf63baf40128b67f89d6a04ef6d","collectionName":"torch.nn.functional.gelu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.hardshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4920a2b542244167ba4d9069551bfdca","collectionName":"torch.nn.functional.hardshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.hardsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d771b201d0cf41bc8281bfe37fbe74af","collectionName":"torch.nn.functional.hardsigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.hardswish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9e5e09bc1e2d4d7f9aeadf03af7559f3","collectionName":"torch.nn.functional.hardswish","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.hardtanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0418877452784810a0b6064e93991384","collectionName":"torch.nn.functional.hardtanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.instance_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ea53d43f13404038b9d621be906dd1ec","collectionName":"torch.nn.functional.instance_norm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.interpolate.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe06d1f2acc14b2ea1f2a4ce4b76f442","collectionName":"torch.nn.functional.interpolate","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.l1_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d64c0a8c1bb14c0f891d277ffbea6927","collectionName":"torch.nn.functional.l1_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.layer_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3f414c4233204f6b9f504ba4eec607cd","collectionName":"torch.nn.functional.layer_norm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.leaky_relu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a3255fc89fdd4821a19d70a2066ec5df","collectionName":"torch.nn.functional.leaky_relu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.leaky_relu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b45b8a63576a46f898ba3d951024173d","collectionName":"torch.nn.functional.leaky_relu_","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.linear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b08227ce0724419d901ee95a4c024f3e","collectionName":"torch.nn.functional.linear","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.local_response_norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8e3a7390441b40dc98fb1a4047babf5b","collectionName":"torch.nn.functional.local_response_norm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.logsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e79b74f2322b4dc3bf710753cbe2824a","collectionName":"torch.nn.functional.logsigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.log_softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5f4c498d33fe4281b055e706aefbb35f","collectionName":"torch.nn.functional.log_softmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.lp_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f0fda0fe57b240fc926bcf734a4962a7","collectionName":"torch.nn.functional.lp_pool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.lp_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d4f3876e91964d1ebd41cf910b2c70b8","collectionName":"torch.nn.functional.lp_pool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.margin_ranking_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e21aaa16206e4bc59eadfe6c7378336c","collectionName":"torch.nn.functional.margin_ranking_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_pool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f08f67d1612248409c5db36f3233ac6c","collectionName":"torch.nn.functional.max_pool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_pool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6bd48e23d4fc4c9d830f07e4754365e5","collectionName":"torch.nn.functional.max_pool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_pool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c5d57fddbe0e4a7898acd56892b956ac","collectionName":"torch.nn.functional.max_pool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_unpool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ec0b13c90198494fbab7cdbec4b57308","collectionName":"torch.nn.functional.max_unpool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_unpool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"32ac590ceaeb41d6b771da5c2908b24e","collectionName":"torch.nn.functional.max_unpool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.max_unpool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"08348ce2d27e456cae911b423855916b","collectionName":"torch.nn.functional.max_unpool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.mse_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"184a6e6f8f3d4ff88551687158c95803","collectionName":"torch.nn.functional.mse_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.multilabel_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c7dd5587ffd54261a3d671188c4c9ed2","collectionName":"torch.nn.functional.multilabel_margin_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.multilabel_soft_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a9434d48e36144a78b8916ab73d8ed3d","collectionName":"torch.nn.functional.multilabel_soft_margin_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.nll_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8d7f9ac3cec84b71bda5afdf9e9924e1","collectionName":"torch.nn.functional.nll_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.normalize.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"becce28932b746989bd7a31896417fc0","collectionName":"torch.nn.functional.normalize","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.pad.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bcb7b3b431194277b2633511bc50a836","collectionName":"torch.nn.functional.pad","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.pairwise_distance.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"86783664627c4840968766ef380b49f5","collectionName":"torch.nn.functional.pairwise_distance","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.pixel_shuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"eedb0410d86f440bbcbaf3bf6107ad20","collectionName":"torch.nn.functional.pixel_shuffle","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.poisson_nll_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba71e2b63cc949cab52527836d269882","collectionName":"torch.nn.functional.poisson_nll_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.prelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"12c46ce88c034463b5c29218a761e896","collectionName":"torch.nn.functional.prelu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.relu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f68e690b61bf47cd9465428f78840851","collectionName":"torch.nn.functional.relu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.relu6.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8382055988504602b5e5126f730422dc","collectionName":"torch.nn.functional.relu6","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.rrelu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"882c56545c684aa5a2e56515691a0d12","collectionName":"torch.nn.functional.rrelu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.rrelu_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cdff83bd428d4b0189fa93d924d7266e","collectionName":"torch.nn.functional.rrelu_","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.selu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7969997e681b41e4b31ef8fe8fcfeb2d","collectionName":"torch.nn.functional.selu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"948e20e0a397423db87e489031a2ede6","collectionName":"torch.nn.functional.sigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.silu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"864f274118444006a32c5142836bda93","collectionName":"torch.nn.functional.silu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ad552bb76764df59e62e2b577c58816","collectionName":"torch.nn.functional.softmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.softmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0a52743edc4d4b8799dbb445f083f039","collectionName":"torch.nn.functional.softmin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.softplus.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cd7e45ff0cc840beaff31b1c2ad889a0","collectionName":"torch.nn.functional.softplus","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.softshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9a993cd8841c4e7ea1b585d9ba690ca2","collectionName":"torch.nn.functional.softshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.softsign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b529146c2b2c48a88a75908a3d947924","collectionName":"torch.nn.functional.softsign","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6c330e8ec9af492681c9cca1f3d3dae6","collectionName":"torch.nn.functional.tanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.tanhshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9260b11624324412992cc2d98ec5e7c1","collectionName":"torch.nn.functional.tanhshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.threshold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b59d2c7a0fbe4e1892521304fdf15983","collectionName":"torch.nn.functional.threshold","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.functional.triplet_margin_loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9249365a1f624f3699d3d6eeea5d1a67","collectionName":"torch.nn.functional.triplet_margin_loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.GELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e2f6875853d14b18a98814980d101c11","collectionName":"torch.nn.GELU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.GroupNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9659cd24bf1f4e98808652c8dc0707ab","collectionName":"torch.nn.GroupNorm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.GRU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1bbbd24753e1439da8033859738af0ba","collectionName":"torch.nn.GRU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.GRUCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b84ef8fe05440518c1de55b1073d5c2","collectionName":"torch.nn.GRUCell","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Hardshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"07a9eebfec2c48f08f9a6b0dc7bb3304","collectionName":"torch.nn.Hardshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Hardsigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4f31d48385474eefb606dc97d44f334b","collectionName":"torch.nn.Hardsigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Hardswish.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9d46c769cea42c781fe3644945d2a95","collectionName":"torch.nn.Hardswish","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Hardtanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"901218e82a364dc38f98a590f2350ba1","collectionName":"torch.nn.Hardtanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Identity.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"936b11b2831842d79addce1df26e829e","collectionName":"torch.nn.Identity","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.InstanceNorm1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3a50f747a0ca458b9b70a7b62f234d8b","collectionName":"torch.nn.InstanceNorm1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.InstanceNorm2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"792eacbc329c46e2b648f83905241418","collectionName":"torch.nn.InstanceNorm2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.InstanceNorm3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"82e5e14261c84fca95b41ad7445ced94","collectionName":"torch.nn.InstanceNorm3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.L1Loss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"56f4043e83e6484e929e7d0e757377e0","collectionName":"torch.nn.L1Loss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LayerNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4159f49202f942929c03908939258389","collectionName":"torch.nn.LayerNorm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LeakyReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe09de59bdfe45858031836d4af56d92","collectionName":"torch.nn.LeakyReLU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Linear.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"96050bed594d457aa14712ff4bb35055","collectionName":"torch.nn.Linear","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LocalResponseNorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ed535b50fdb644b389dde24bc8819d71","collectionName":"torch.nn.LocalResponseNorm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LogSigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac354aa695c54ee8bc0933acc74b301e","collectionName":"torch.nn.LogSigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LogSoftmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d5ff9413b2504372962565b7eee3e6b0","collectionName":"torch.nn.LogSoftmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LPPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"563e90b5bbd645778ebf9a1a0c51f2b7","collectionName":"torch.nn.LPPool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LPPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2b4b42ac574747df963e8628bb7a015c","collectionName":"torch.nn.LPPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LSTM.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17720b28e9e44707a10fcde6ed3161cd","collectionName":"torch.nn.LSTM","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.LSTMCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fc10d8e690164fb4a9b9d1504ae24be1","collectionName":"torch.nn.LSTMCell","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MarginRankingLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"78be3f3a8481425682e27c46687d6673","collectionName":"torch.nn.MarginRankingLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxPool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1b6d305ed42e406e945dac3bb0e08620","collectionName":"torch.nn.MaxPool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxPool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"72043149e6e449358dabee2f6a60ca2f","collectionName":"torch.nn.MaxPool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxPool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"36d923b5e6ad44e4b6a825e2d71e1d27","collectionName":"torch.nn.MaxPool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxUnpool1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2f4fccf5a643429a91fe35a3fbc997c9","collectionName":"torch.nn.MaxUnpool1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxUnpool2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e63342b1cade4f378295999518f73658","collectionName":"torch.nn.MaxUnpool2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MaxUnpool3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"442ee1cb2f0348efa385c8bca6c80704","collectionName":"torch.nn.MaxUnpool3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MSELoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e5948e696b7c48f397ffbabd3bbd2fa1","collectionName":"torch.nn.MSELoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MultiheadAttention.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"214831cfdc4a44fea4953c82a7c30209","collectionName":"torch.nn.MultiheadAttention","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MultiLabelMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c206f9fca81b4dce8b7abba16792ebae","collectionName":"torch.nn.MultiLabelMarginLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.MultiLabelSoftMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"244a453c35134f4eb75ccba291edeec8","collectionName":"torch.nn.MultiLabelSoftMarginLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.NLLLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"08c262619f1f4189b33c7323443620cd","collectionName":"torch.nn.NLLLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.PairwiseDistance.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4226aa961c6f4361a17fbf1100f8afac","collectionName":"torch.nn.PairwiseDistance","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.PixelShuffle.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"235617ae04c04185be6376cd0ffcd54b","collectionName":"torch.nn.PixelShuffle","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.PoissonNLLLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"981e75f074014e9b80fdcd1c385b4df3","collectionName":"torch.nn.PoissonNLLLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.PReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a5e2ea6dbfbd433b9a3300949efc43a3","collectionName":"torch.nn.PReLU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReflectionPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"046af69aa29c424c87b18be46ea25f1e","collectionName":"torch.nn.ReflectionPad1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReflectionPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"35e0dc47d4a74594a30247598d8a6c6f","collectionName":"torch.nn.ReflectionPad2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"88e07d278b3448758110b316c9e9af5a","collectionName":"torch.nn.ReLU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReLU6.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1887f02ed6694b19bc0b10a995e11f12","collectionName":"torch.nn.ReLU6","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReplicationPad1d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"896ef42087b24fdd93f8c6a8e0f534fd","collectionName":"torch.nn.ReplicationPad1d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReplicationPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9cd5b9053b88409899bad806d1549ebe","collectionName":"torch.nn.ReplicationPad2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ReplicationPad3d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6bd49794d338476ba7368ac46e998388","collectionName":"torch.nn.ReplicationPad3d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.RNN.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7e28f5451b164d79b15ccb58f829ec4b","collectionName":"torch.nn.RNN","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.RNNBase.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ff7355ef78824bf4b5cec23b63e56215","collectionName":"torch.nn.RNNBase","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.RNNCell.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"388a7cc8d27c44ba9174d6e83cf5d6bb","collectionName":"torch.nn.RNNCell","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.RReLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6dd33d78ce4a4a4585effdebf72fee5d","collectionName":"torch.nn.RReLU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.SELU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9b1e4e1bb83c45eaa95823dd4b531107","collectionName":"torch.nn.SELU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"33478c87a0754d54b152ce3b8e31922b","collectionName":"torch.nn.Sigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.SiLU.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9ca1e878bb3f4d31967a82ba497ae998","collectionName":"torch.nn.SiLU","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softmax.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f78b43cb65084aed96eea41aa71c57f1","collectionName":"torch.nn.Softmax","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softmax2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6d7aebe93d054f9d92083fc4a122d7bb","collectionName":"torch.nn.Softmax2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softmin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3496c11a682f461a9ca96b583efd40d0","collectionName":"torch.nn.Softmin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softplus.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b4f4b8b07d4c490f8a2bc8ab9a442d33","collectionName":"torch.nn.Softplus","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"63b08cf40eb0429db4fea625ca982848","collectionName":"torch.nn.Softshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Softsign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ee3df9d8e8f34ee8828e5b4754ae7ab7","collectionName":"torch.nn.Softsign","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cf807a6549e14484a2a1bafdc7da3000","collectionName":"torch.nn.Tanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Tanhshrink.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"154552c83b4d47c5823f74421ac0986a","collectionName":"torch.nn.Tanhshrink","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Threshold.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fe5c69b2e8684960a42344d3a0f7cad2","collectionName":"torch.nn.Threshold","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.TransformerDecoderLayer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9f033ac7f7da45e4a71e3f5366b687ed","collectionName":"torch.nn.TransformerDecoderLayer","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.TransformerEncoderLayer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7f30e0c71b7f46d9ad3b44fa5a19d004","collectionName":"torch.nn.TransformerEncoderLayer","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.TripletMarginLoss.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dddf859c4f8049888f32c56ba0056c27","collectionName":"torch.nn.TripletMarginLoss","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.Upsample.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a81a1dc438940ad8f403d4e5f4c2b1d","collectionName":"torch.nn.Upsample","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.UpsamplingBilinear2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9f12519a03c4c3c996245d214422657","collectionName":"torch.nn.UpsamplingBilinear2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.UpsamplingNearest2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"0ba3454e1730487faf2682787dabf93b","collectionName":"torch.nn.UpsamplingNearest2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.utils.clip_grad_norm_.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"366a9c036d1d4e73b1517ce88eb862e8","collectionName":"torch.nn.utils.clip_grad_norm_","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.utils.rnn.pad_sequence.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"09de55f90b654dcdb5d2280b475980d4","collectionName":"torch.nn.utils.rnn.pad_sequence","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nn.ZeroPad2d.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"323346d5f5284c0a819ac41eb8354fa8","collectionName":"torch.nn.ZeroPad2d","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.nonzero.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2a4d659532f74b8db5de76914c74fc97","collectionName":"torch.nonzero","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.norm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5140248202db4de19c14a157f005706d","collectionName":"torch.norm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.normal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b77b6386f5b54058988d795a4eb1e3c0","collectionName":"torch.normal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.numel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2f329af83f234132b8623105ce9f95a8","collectionName":"torch.numel","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ones.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d1d50784c50545bd8b171f44a9fc6473","collectionName":"torch.ones","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ones_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5f90fd6a07134cf0b74f440afbdc6f1e","collectionName":"torch.ones_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.outer.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad8865f4986240b6b369faba388b8851","collectionName":"torch.outer","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.pinverse.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8522c697f368498da0bed88eecbdd7ef","collectionName":"torch.pinverse","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.poisson.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1d29b556b32e430c9b9d4a4f2871b498","collectionName":"torch.poisson","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.polar.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"695fd926cb764828b584af7baba60e7e","collectionName":"torch.polar","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.polygamma.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e755fa9bfa1a406a89fe5e89380bbce1","collectionName":"torch.polygamma","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.pow.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"afe30980333a44828583330e9227bf91","collectionName":"torch.pow","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.prod.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9b4616de60fa462792afb60101221394","collectionName":"torch.prod","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.promote_types.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7f00b77c47ab425298fc3975fd21e745","collectionName":"torch.promote_types","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.qr.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8aaf2384a2b94abcb20f6c8f939f4838","collectionName":"torch.qr","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.quantile.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1b32a15048184aa2a4d5ab3d46fc9f14","collectionName":"torch.quantile","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.rad2deg.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f49837d0f1db4bbc953b0b3f520f93f2","collectionName":"torch.rad2deg","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.rand.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4cbba803ba8c45d5a59dafccb080c4ac","collectionName":"torch.rand","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.randint.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6fb7b7ab6eb04850aab759f2cbe09188","collectionName":"torch.randint","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.randn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3d901728976d42318713a62508524380","collectionName":"torch.randn","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.randn_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e9acfdb8b72a4b3d93db04d995e85eac","collectionName":"torch.randn_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.randperm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"83f0a74f8aed4c29ae5149483e62693d","collectionName":"torch.randperm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.rand_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ac225eb2d49a4a8485144093a0e9e229","collectionName":"torch.rand_like","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.range.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"a9393bb19378498b84c0c79c016b829c","collectionName":"torch.range","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.ravel.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"40ac2e38b5144d98878be1367fb2b76a","collectionName":"torch.ravel","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.real.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ad154b44e3314d8f8b6c482376ef7642","collectionName":"torch.real","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.reciprocal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8b639e34f797407c9fbe62014d7b09c2","collectionName":"torch.reciprocal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.remainder.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c8a57e99165a4dee921aa6578c9cdcb6","collectionName":"torch.remainder","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.renorm.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"57673ba7e4a94cd4869fa869e4ef80e4","collectionName":"torch.renorm","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.repeat_interleave.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"137ee8a2734442afab7f95d9550788b5","collectionName":"torch.repeat_interleave","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.reshape.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"883e3d6f5a764e1c8ac2710dc423a7d0","collectionName":"torch.reshape","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.result_type.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2d34f0acd5d64a84abc00017a120336a","collectionName":"torch.result_type","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.roll.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"67a12016c160404498e6e53963a513c0","collectionName":"torch.roll","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.rot90.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"413c71ef9350461095f7ab721a2ae44d","collectionName":"torch.rot90","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.round.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"115b7c535da04e20b61e9bfde1c095b8","collectionName":"torch.round","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.rsqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1eb77b6d8bc646729446c41bdd548f0e","collectionName":"torch.rsqrt","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.save.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7dd773f7595e4c089bd537354ef85b8f","collectionName":"torch.save","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.scatter.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1d351ce3dee64aa18622d66e232add83","collectionName":"torch.scatter","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.scatter_add.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b9cf0dd79efe4d31a89b145902dd65e2","collectionName":"torch.scatter_add","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.searchsorted.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c8e27c71bb9a49b5a1058e78eb18ab4c","collectionName":"torch.searchsorted","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.set_default_dtype.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"fbad810916d843689244c3d05cecd52c","collectionName":"torch.set_default_dtype","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.set_flush_denormal.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1b002377f66a4b0eac3c1a06c094292a","collectionName":"torch.set_flush_denormal","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.set_num_threads.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ecb1e82918aa422fa18661c777810d25","collectionName":"torch.set_num_threads","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sgn.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c6974a36bae34e5d9f19d6b50bdf5f1e","collectionName":"torch.sgn","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sigmoid.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8eaa882ec2de42079ba3b13c10d81f14","collectionName":"torch.sigmoid","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sign.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dec5cd18bb9d45cb92e7b9b641bdc89f","collectionName":"torch.sign","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.signbit.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"e331a9a419c848df8cbf9bfd8f1b5016","collectionName":"torch.signbit","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sin.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"b94e85004908448d9e9e0b1fa748d1fc","collectionName":"torch.sin","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sinh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"215174c3f3b54c689821c374c5f54069","collectionName":"torch.sinh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.slogdet.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"10e2cbf05ed147bb829413b3e3e2e4ad","collectionName":"torch.slogdet","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6425c1c1ffc347dd83e871305e664601","collectionName":"torch.solve","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sort.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3b5d85312732445bb26c20a8a95cef06","collectionName":"torch.sort","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sparse_coo_tensor.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"bffafadc5ceb4f35bb019f6377dc45f0","collectionName":"torch.sparse_coo_tensor","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"17fd859854d043d1b847b6e8fbe3b41b","collectionName":"torch.split","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sqrt.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7b1eacb276fc49aab6858018a11a890a","collectionName":"torch.sqrt","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.square.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8f5d34f355434c4fb08993b5ddad8ad5","collectionName":"torch.square","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.squeeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"747ef5064f114683bfe538a5b5d5839b","collectionName":"torch.squeeze","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.stack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5f129078975643419781e71684dc431e","collectionName":"torch.stack","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.std.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5a263e08f8b24ec39143d17a10f0f053","collectionName":"torch.std","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.std_mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c415a01cf2a54cb78fd592e6f15fbd93","collectionName":"torch.std_mean","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sub.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c1c24deffaaf489db82414f30ca190e6","collectionName":"torch.sub","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.sum.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3bedf67f0eea40998fd32d6d63aca958","collectionName":"torch.sum","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.svd.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1e51758249a645508b3bca6075104ab6","collectionName":"torch.svd","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.swapaxes.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2a43e0355a354ce49d7201e9c9d0252d","collectionName":"torch.swapaxes","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.swapdims.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c9e1a7ad0d9744a6964a3df95e9039c9","collectionName":"torch.swapdims","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.symeig.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7302ee4b28be4f268b1fcdb2524d640d","collectionName":"torch.symeig","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.t.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5169332a9acd41febbddd76e52658e14","collectionName":"torch.t","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.take.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"581d8dbd3e7d4e47a8962b9801f43aec","collectionName":"torch.take","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tan.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6c0ea35deb234b3798691afa7e1b52be","collectionName":"torch.tan","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tanh.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6dd4349412ef417f86bd71ce68c61dba","collectionName":"torch.tanh","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tensordot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"6d0f670f076848fd94e4dc47af6b113f","collectionName":"torch.tensordot","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tensor_split.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"7a3bdd37a0c4494082a4e9b11ed136ed","collectionName":"torch.tensor_split","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.topk.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"783bdc0f0cfc4f49a7a2db3e32f36103","collectionName":"torch.topk","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.trace.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1be77fc848c74f8d90f357f2a250c9f6","collectionName":"torch.trace","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.transpose.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"9a1426590a0e4ee986755ad3db656f90","collectionName":"torch.transpose","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.trapz.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"dca73c9b6eb74c9d8c08fe962262e294","collectionName":"torch.trapz","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.triangular_solve.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"2ce3234a23ee43a8ac366d5b0bbe0766","collectionName":"torch.triangular_solve","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tril.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"cee2ce5dc04643f584c1717c5a1b95e7","collectionName":"torch.tril","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.tril_indices.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ede1b85224a24f359541a4c1c011f660","collectionName":"torch.tril_indices","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.triu.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"835a45c65967463692d26c6556512b74","collectionName":"torch.triu","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.triu_indices.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"966c5de0f81a4c8d94e6943d9d35d34c","collectionName":"torch.triu_indices","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.true_divide.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c62fa253101f4e548e16ab1a90e1200d","collectionName":"torch.true_divide","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.trunc.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"23bfb0c256b640ec9e154f62406fe2a4","collectionName":"torch.trunc","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.unbind.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"f0510256a4b54d4aa1c37ccf0c42849d","collectionName":"torch.unbind","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.unique.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d9b2efeaaa654ea896476a02cb623dd3","collectionName":"torch.unique","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.unique_consecutive.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"d47f15d4fef24fa99ffa17dff76c129b","collectionName":"torch.unique_consecutive","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.unsqueeze.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"1815acda2d9d4f4789ffc9593a71b75e","collectionName":"torch.unsqueeze","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.use_deterministic_algorithms.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"21b24cff68654f0386c7b8f132422071","collectionName":"torch.use_deterministic_algorithms","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.vander.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"ba5d8dc7d39c418eb9ce08cdba18c3e1","collectionName":"torch.vander","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.var.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5611c093a281403aa8b076f6858d41cb","collectionName":"torch.var","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.var_mean.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"3128958138564152990273d0a2d5d0d2","collectionName":"torch.var_mean","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.vdot.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"83484b12223448d7bd7ef1050b4dca28","collectionName":"torch.vdot","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.view_as_complex.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"8df61b5056974c00a08924ac8ae0bd9b","collectionName":"torch.view_as_complex","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.view_as_real.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"798043de11994ff8944832f73a3bd012","collectionName":"torch.view_as_real","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.vstack.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4a6d96d083894bc9aad1e9eec1f5d391","collectionName":"torch.vstack","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.where.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"48fda248713b4d509c4d46aaf8bf0838","collectionName":"torch.where","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.xlogy.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"5df33b536bf74af9a1c9dbeddbaedfd1","collectionName":"torch.xlogy","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.zeros.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"c36ee1e4a9124a218203ca31c66daef1","collectionName":"torch.zeros","type":"collection"}

================================================================================
FILE: dump\freefuzz-torch\torch.zeros_like.metadata.json
================================================================================
{"indexes":[{"v":{"$numberInt":"2"},"key":{"_id":{"$numberInt":"1"}},"name":"_id_"}],"uuid":"4b8226a951604f1fa4dbae95409da275","collectionName":"torch.zeros_like","type":"collection"}

================================================================================
FILE: input_models\pytorch_models.csv
================================================================================
Link,# models
https://github.com/weiaicunzai/pytorch-cifar100,30
https://github.com/pytorch/examples/tree/master/super_resolution,1
https://github.com/yiyang7/Super_Resolution_with_CNNs_and_GANs,4
https://github.com/JaidedAI/EasyOCR,2
https://github.com/lucidrains/stylegan2-pytorch,1
https://github.com/lucidrains/stylegan2-pytorch,1
https://github.com/pytorch/examples/tree/master/fast_neural_style,1
https://github.com/wvangansbeke/LaneDetection_End2End,1
https://github.com/maudzung/RTM3D,2
https://github.com/cgraber/cvpr_dNRI,1
https://github.com/vincent-leguen/DILATE,1
https://github.com/lonePatient/Bert-Multi-Label-Text-Classification,1
https://huggingface.co/models,4
https://github.com/huggingface/transformers,11
https://github.com/thu-ml/tianshou/tree/master/examples/atari,10
https://github.com/znxlwm/pytorch-generative-model-collections,10
https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph,11
https://github.com/JavierAntoran/Bayesian-Neural-Networks,7
https://github.com/Mariewelt/OpenChem,1
https://github.com/Megvii-Nanjing/ML-GCN,1
https://github.com/lehaifeng/T-GCN,1
total,102


================================================================================
FILE: src\FreeFuzz.py
================================================================================
from utils.skip import need_skip_torch
import configparser
from os.path import join
import subprocess
from utils.printer import dump_data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeFuzz: a fuzzing frameword for deep learning library")
    parser.add_argument("--conf", type=str, default="demo.conf", help="configuration file")
    args = parser.parse_args()

    config_name = args.conf
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join(__file__.replace("FreeFuzz.py", "config"), config_name))

    libs = freefuzz_cfg["general"]["libs"].split(",")
    print("Testing on ", libs)
    
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    # output configuration
    output_cfg = freefuzz_cfg["output"]
    torch_output_dir = output_cfg["torch_output"]
    tf_output_dir = output_cfg["tf_output"]

    if "torch" in libs:
        # database configuration
        from classes.database import TorchDatabase
        TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])

        for api_name in TorchDatabase.get_api_list():
            print(api_name)
            if need_skip_torch(api_name):
                continue
            try:
                res = subprocess.run(["python3", "FreeFuzz_api.py", config_name, "torch", api_name], shell=False, timeout=100)
            except subprocess.TimeoutExpired:
                dump_data(f"{api_name}\n", join(torch_output_dir, "timeout.txt"), "a")
            except Exception as e:
                dump_data(f"{api_name}\n  {e}\n", join(torch_output_dir, "runerror.txt"), "a")
            else:
                if res.returncode != 0:
                    dump_data(f"{api_name}\n", join(torch_output_dir, "runcrash.txt"), "a")
    if "tf" in libs:
        # database configuration
        from classes.database import TFDatabase
        TFDatabase.database_config(host, port, mongo_cfg["tf_database"])

        for api_name in TFDatabase.get_api_list():
            print(api_name)
            try:
                res = subprocess.run(["python3", "FreeFuzz_api.py", config_name, "tf", api_name], shell=False, timeout=100)
            except subprocess.TimeoutExpired:
                dump_data(f"{api_name}\n", join(tf_output_dir, "timeout.txt"), "a")
            except Exception as e:
                dump_data(f"{api_name}\n  {e}\n", join(tf_output_dir, "runerror.txt"), "a")
            else:
                if res.returncode != 0:
                    dump_data(f"{api_name}\n", join(tf_output_dir, "runcrash.txt"), "a")
    
    not_test = []
    for l in libs:
        if l not in ["tf", "torch"]: not_test.append(l)
    if len(not_test):
        print(f"WE DO NOT SUPPORT SUCH DL LIBRARY: {not_test}!")


================================================================================
FILE: src\FreeFuzz_api.py
================================================================================
import sys
from constants.enum import OracleType
import configparser
from os.path import join
from utils.converter import str_to_bool


if __name__ == "__main__":
    config_name = sys.argv[1]
    library = sys.argv[2]
    api_name = sys.argv[3]

    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join(__file__.replace("FreeFuzz_api.py", "config"), config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    # oracle configuration
    oracle_cfg = freefuzz_cfg["oracle"]
    crash_oracle = str_to_bool(oracle_cfg["enable_crash"])
    cuda_oracle = str_to_bool(oracle_cfg["enable_cuda"])
    precision_oracle = str_to_bool(oracle_cfg["enable_precision"])

    diff_bound = float(oracle_cfg["float_difference_bound"])
    time_bound = float(oracle_cfg["max_time_bound"])
    time_thresold = float(oracle_cfg["time_thresold"])

    # output configuration
    output_cfg = freefuzz_cfg["output"]
    torch_output_dir = output_cfg["torch_output"]
    tf_output_dir = output_cfg["tf_output"]

    # mutation configuration
    mutation_cfg = freefuzz_cfg["mutation"]
    enable_value = str_to_bool(mutation_cfg["enable_value_mutation"])
    enable_type = str_to_bool(mutation_cfg["enable_type_mutation"])
    enable_db = str_to_bool(mutation_cfg["enable_db_mutation"])
    each_api_run_times = int(mutation_cfg["each_api_run_times"])

    if library.lower() in ["pytorch", "torch"]:
        import torch
        from classes.torch_library import TorchLibrary
        from classes.torch_api import TorchAPI
        from classes.database import TorchDatabase
        from utils.skip import need_skip_torch

        TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])

        if cuda_oracle and not torch.cuda.is_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        # Pytorch TEST
        MyTorch = TorchLibrary(torch_output_dir, diff_bound, time_bound,
                            time_thresold)
        for _ in range(each_api_run_times):
            api = TorchAPI(api_name)
            api.mutate(enable_value, enable_type, enable_db)
            if crash_oracle:
                MyTorch.test_with_oracle(api, OracleType.CRASH)
            if cuda_oracle:
                MyTorch.test_with_oracle(api, OracleType.CUDA)
            if precision_oracle:
                MyTorch.test_with_oracle(api, OracleType.PRECISION)
    elif library.lower() in ["tensorflow", "tf"]:
        import tensorflow as tf
        from classes.tf_library import TFLibrary
        from classes.tf_api import TFAPI
        from classes.database import TFDatabase
        from utils.skip import need_skip_tf

        TFDatabase.database_config(host, port, mongo_cfg["tf_database"])
        if cuda_oracle and not tf.test.is_gpu_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        
        MyTF = TFLibrary(tf_output_dir, diff_bound, time_bound,
                            time_thresold)
        print(api_name)
        if need_skip_tf(api_name): pass
        else:
            for _ in range(each_api_run_times):
                api = TFAPI(api_name)
                api.mutate(enable_value, enable_type, enable_db)
                if crash_oracle:
                    MyTF.test_with_oracle(api, OracleType.CRASH)
                if cuda_oracle:
                    MyTF.test_with_oracle(api, OracleType.CUDA)
                if precision_oracle:
                    MyTF.test_with_oracle(api, OracleType.PRECISION)
    else:
        print(f"WE DO NOT SUPPORT SUCH DL LIBRARY: {library}!")


================================================================================
FILE: src\classes\api.py
================================================================================
import inspect
from numpy.random import randint, choice
from classes.argument import ArgType, Argument, OracleType
from utils.probability import *



class API:
    def __init__(self, api_name):
        self.api = api_name

    def mutate(self):
        pass

    def to_code(self) -> str:
        pass

    def to_dict(self) -> dict:
        pass

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        pass

    @staticmethod
    def indent_code(code):
        codes = code.split("\n")
        result = []
        for code in codes:
            if code == "":
                continue
            result.append("  " + code)
        return "\n".join(result) + "\n"



================================================================================
FILE: src\classes\argument.py
================================================================================
from numpy.random import choice, randint
from enum import IntEnum
from utils.probability import *
from constants.enum import OracleType

class ArgType(IntEnum):
    INT = 1
    STR = 2
    FLOAT = 3
    BOOL = 4
    TUPLE = 5
    LIST = 6
    NULL = 7
    TORCH_OBJECT = 8
    TORCH_TENSOR = 9
    TORCH_DTYPE = 10
    TF_TENSOR = 11
    TF_DTYPE = 12
    KERAS_TENSOR = 13
    TF_VARIABLE = 14
    TF_OBJECT = 15
    



class Argument:
    """
    _support_types: all the types that Argument supports.
    NOTICE: The inherent class should call the method of its parent
    when it does not support its type
    """
    _support_types = [
        ArgType.INT, ArgType.STR, ArgType.FLOAT, ArgType.NULL, ArgType.TUPLE,
        ArgType.LIST, ArgType.BOOL
    ]
    _int_values = [-1024, -16, -1, 0, 1, 16, 1024]
    _str_values = [
        "mean", "sum", "max", 'zeros', 'reflect', 'circular', 'replicate'
    ]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0, 1024.0, -1024.0, 1e20, -1e20]

    def __init__(self, value, type: ArgType):
        self.value = value
        self.type = type

    def to_code(self, var_name: str) -> str:
        """ArgType.LIST and ArgType.TUPLE should be converted to code in the inherent class"""
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.STR:
            return f"{var_name} = \"{self.value}\"\n"
        elif self.type == ArgType.NULL:
            return f"{var_name} = None\n"
        else:
            assert (0)

    def mutate_value(self) -> None:
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = not self.value
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # self.value is a list now
            for arg in self.value:
                arg.mutate_value()
        elif self.type == ArgType.NULL:
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        """The type mutation for NULL should be implemented in the inherent class"""
        if self.type in [
                ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL
        ]:
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("max")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            for arg in self.value:
                arg.mutate_type()
        else:
            # cannot change the type of assert in the general Argument
            assert (0)


    def mutate_int_value(self, value, _min=None, _max=None) -> int:
        if choose_from_list():
            value = choice(Argument._int_values)
        else:
            value += randint(-64, 64)
        # min <= value <= max
        if _min != None:
            value = max(_min, value)
        if _max != None:
            value = min(_max, value)
        return value


    def mutate_str_value(self, value) -> str:
        """You can add more string mutation strategies"""
        if choose_from_list():
            return choice(Argument._str_values)
        else:
            return value


    def mutate_float_value(self, value) -> float:
        if choose_from_list():
            return choice(Argument._float_values)
        else:
            return value + randint(-64, 64) * 1.0


    def initial_value(self, type: ArgType):
        """LIST and TUPLE should be implemented in the inherent class"""
        if type == ArgType.INT:
            return choice(Argument._int_values)
        elif type == ArgType.FLOAT:
            return choice(Argument._float_values)
        elif type == ArgType.STR:
            return choice(Argument._str_values)
        elif type == ArgType.BOOL:
            return choice([True, False])
        elif type == ArgType.NULL:
            return None
        else:
            assert (0)
    
    @staticmethod
    def get_type(x):
        if x is None:
            return ArgType.NULL
        elif isinstance(x, bool):
            return ArgType.BOOL
        elif isinstance(x, int):
            return ArgType.INT
        elif isinstance(x, str):
            return ArgType.STR
        elif isinstance(x, float):
            return ArgType.FLOAT
        elif isinstance(x, tuple):
            return ArgType.TUPLE
        elif isinstance(x, list):
            return ArgType.LIST
        else:
            return None



================================================================================
FILE: src\classes\database.py
================================================================================
import pymongo
from numpy.random import choice
"""
This file is the interfere with database
"""

class Database:
    """Database setting"""
    signature_collection = "signature"
    similarity_collection = "similarity"
    argdef_collection = "api_args"

    def __init__(self) -> None:
        pass

    def database_config(self, host, port, database_name):
        self.DB = pymongo.MongoClient(host=host, port=port)[database_name]

    def index_name(self, api_name, arg_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name})
        if record == None:
            print(f"No such {api_name}")
            return None
        arg_names = record["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    def select_rand_over_db(self, api_name, arg_name):
        if api_name not in self.DB.list_collection_names():
            return None, False
        arg_names = self.DB[self.signature_collection].find_one({"api": api_name})["args"]
        if arg_name.startswith("parameter:"):
            index = int(arg_name[10:])
            if index >= len(arg_names):
                return None, False
            arg_name = arg_names[index]

        sim_dict = self.DB[self.similarity_collection].find_one({
            "api": api_name,
            "arg": arg_name
        })
        if sim_dict == None:
            return None, False
        APIs = sim_dict["APIs"]
        probs = sim_dict["probs"]
        if len(APIs) == 0:
            return None, False
        target_api = choice(APIs, p=probs)
        # compare the time of 2 operations
        idx_name = self.index_name(target_api, arg_name)
        if idx_name == None:
            return None, False
        select_data = self.DB[target_api].aggregate([{
            "$match": {
                "$or": [{
                    arg_name: {
                        "$exists": True
                    },
                }, {
                    idx_name: {
                        "$exists": True
                    }
                }]
            }
        }, {
            "$sample": {
                "size": 1
            }
        }])
        if not select_data.alive:
            # not found any value in the (target_api, arg_name)
            print(f"ERROR IN SIMILARITY: {target_api}, {api_name}")
            return None, False
        select_data = select_data.next()
        if arg_name in select_data.keys():
            return select_data[arg_name], True
        else:
            return select_data[idx_name], True


    def get_rand_record(self, api_name):
        record = self.DB[api_name].aggregate([{"$sample": {"size": 1}}])
        if not record.alive:
            print(f"NO SUCH API: {api_name}")
            assert(0)
        record = record.next()
        record.pop("_id")
        assert("_id" not in record.keys())
        return record
    
    def get_all_records(self, api_name):
        if api_name not in self.DB.list_collection_names():
            print(f"NO SUCH API: {api_name}")
            return []
        temp = self.DB[api_name].find({}, {"_id": 0})
        records = []
        for t in temp:
            assert("_id" not in t.keys())
            records.append(t)
        return records
    
    def get_signature(self, api_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name}, {"_id": 0})
        if record == None:
            print(f"NO SIGNATURE FOR: {api_name}")
            assert(0)
        return record["args"]

    @staticmethod
    def get_api_list(DB, start_str):
        api_list = []
        for name in DB.list_collection_names():
            if name.startswith(start_str):
                api_list.append(name)
        return api_list

class TorchDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "torch.")
        return self.api_list

class TFDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "tf.")
        return self.api_list
"""
Database for each library
NOTE:
You must config the database by using `database_config(host, port, name)` before use!!!
Like TFDatabase.database_config("127.0.0.1", 27109, "tftest")
"""
TorchDatabase = TorchDB()
TFDatabase = TFDB()

================================================================================
FILE: src\classes\library.py
================================================================================
from classes.argument import *
from classes.api import *
from os.path import join
import os

class Library:
    def __init__(self, directory) -> None:
        def init_dir(dir_name):
            os.makedirs(join(dir_name, "success"), exist_ok=True)
            os.makedirs(join(dir_name, "potential-bug"), exist_ok=True)
            os.makedirs(join(dir_name, "fail"), exist_ok=True)
            os.makedirs(join(dir_name, "compare-bug"), exist_ok=True)

        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.output = {
            OracleType.CRASH: join(directory, "crash-oracle"),
            OracleType.CUDA: join(directory, "cuda-oracle"),
            OracleType.PRECISION: join(directory, "precision-oracle"),
        }
        for dir_name in self.output.values():
            init_dir(dir_name)
    
    @staticmethod
    def generate_code():
        pass

    @staticmethod
    def write_to_dir(dir, api_name, code):
        api_dir = join(dir, api_name)
        if not os.path.exists(api_dir):
            os.makedirs(api_dir)
        filenames = os.listdir(api_dir)
        max_name = 0
        for name in filenames:
            max_name = max(max_name, int(name.replace(".py", "")))
        new_name = str(max_name + 1)
        with open(join(api_dir, new_name + ".py"), "w") as f:
            f.write(code)


================================================================================
FILE: src\classes\tf_api.py
================================================================================
from functools import WRAPPER_UPDATES
import inspect
import json
import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from numpy.random import choice, randint

from constants.keys import *
from classes.argument import ArgType, Argument
from classes.api import API
from termcolor import colored

from classes.api import API
from classes.database import TFDatabase

from classes.argument import OracleType
from utils.probability import do_type_mutation, do_select_from_db

class TFArgument(Argument):
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TF_TENSOR, ArgType.KERAS_TENSOR, ArgType.TF_VARIABLE]
    _dtypes = [
        tf.bfloat16, tf.bool, tf.complex128, tf.complex64, tf.double,
        tf.float16, tf.float32, tf.float64, tf.half,
        tf.int16, tf.int32, tf.int64, tf.int8,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
    ]
    _support_types = [
        ArgType.TF_TENSOR, ArgType.TF_VARIABLE, ArgType.KERAS_TENSOR,
        ArgType.TF_DTYPE, ArgType.TF_OBJECT
    ]

    def __init__(self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None) -> None:
        if isinstance(dtype, str):
            dtype = self.str_to_dtype(dtype)
        shape = self.shape_to_list(shape)

        super().__init__(value, type)
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    @staticmethod
    def shape_to_list(shape): 
        if shape is None: return None   
        if not isinstance(shape, list):
            try:
                shape = shape.as_list()
            except:
                shape = list(shape)
            else:
                shape = list(shape)
        shape = [1 if x is None else x for x in shape]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if tf.is_tensor(x):
            if tf.keras.backend.is_keras_tensor(x):
                return ArgType.KERAS_TENSOR
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        
    def mutate_value_random(self) -> None:
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            self.minv, self.maxv = self.random_tensor_value_range(self.dtype)
        elif self.type == ArgType.TF_DTYPE:
            self.value = TFArgument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)
            assert (0)

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = self.mutate_int_value(new_shape[i], minv=0)
               
        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(0.)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [TFArgument(1, ArgType.INT), TFArgument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1, 3), randint(1, 3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.1

        def if_mutate_null():
            return random.random() < 0.1

        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive(): return False
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            if random.random() < 0.01: 
                self.value = [] # with a probability return an empty list
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TF_TENSOR:
            dtype = choice(self._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            if not if_mutate_null():
                return False
            new_type = choice(self._support_types + super()._support_types)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TFArgument(2, ArgType.INT),
                    TFArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TF_TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(TFArgument._dtypes)
        return True

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.2

    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1

    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.2

    
    def mutate_bool_value(self, value) -> bool:
        return choice([True, False])

    def mutate_int_value(self, value, minv=None, maxv=None) -> int:
        if TFArgument.if_mutate_int_random():
            value = choice(self._int_values)
        else:
            value += randint(-2, 2)
        if minv is not None:
            value = max(minv, value)
        if maxv is not None:
            value = min(maxv, value)
        return value
    
    def mutate_str_value(self, value) -> str:
        if TFArgument.if_mutate_str_random():
            return choice(self._str_values)
        return value

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(TFArgument._dtypes)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [tf.int16, tf.int32, tf.int64]:
            return tf.int8
        elif dtype in [tf.float32, tf.float64]:
            return tf.float16
        elif dtype in [ tf.complex128]:
            return tf.complex64
        return dtype

    @staticmethod
    def random_tensor_value_range(dtype):
        assert isinstance(dtype, tf.dtypes.DType)
        minv = 0
        maxv = 1
        if dtype.is_floating or dtype.is_complex or dtype == tf.string or dtype == tf.bool:
            pass
        elif "int64" in dtype.name or "int32" in dtype.name or "int16" in dtype.name:
            minv = 0 if "uint" in dtype.name else - (1 << 8)
            maxv = (1 << 8)
        else:
            try:
                minv = dtype.min
                maxv = dtype.max
            except Exception as e:
                minv, maxv = 0, 1
        return minv, maxv

    def to_code_tensor(self, var_name, low_precision=False):
        dtype = self.dtype
        if low_precision:
            dtype = self.low_precision_dtype(dtype)
        shape = self.shape
        if dtype is None:
            assert (0)
        code = ""
        var_tensor_name = f"{var_name}_tensor"
        if dtype.is_floating:
            code += "%s = tf.random.uniform(%s, dtype=tf.%s)\n" % (var_tensor_name, shape, dtype.name)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            code += "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s)," \
                    "tf.random.uniform(%s, dtype=tf.%s))\n" % (var_tensor_name, shape, ftype, shape, ftype)
        elif dtype == tf.bool:
            code += "%s = tf.cast(tf.random.uniform(" \
                   "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (var_tensor_name, shape)
        elif dtype == tf.string:
            code += "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (var_tensor_name, shape)
        elif dtype in [tf.int32, tf.int64]:
            code += "%s = tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.%s)\n" \
                % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        else:
            code += "%s = tf.saturate_cast(" \
                "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                "dtype=tf.%s)\n" % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        code += f"{var_name} = tf.identity({var_tensor_name})\n"
        return code

    def to_code_keras_tensor(self, var_name, low_precision=False):
        return self.to_code_tensor(var_name, low_precision=low_precision)

    def to_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            # Did not consider cloning for in-place operation here.
            code = ""
            if self.type == ArgType.TF_TENSOR:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
            elif self.type == ArgType.TF_VARIABLE:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            elif self.type == ArgType.KERAS_TENSOR:
                code = self.to_code_keras_tensor(var_name, low_precision=low_precision)
            return code
        return super().to_code(var_name)


    def to_diff_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            code = f"{var_name} = tf.identity({var_name}_tensor)\n"
            if not low_precision:
                code += f"{var_name} = tf.cast({var_name}, tf.{self.dtype.name})\n"
            if self.type == ArgType.TF_VARIABLE:
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            return code
        return ""

    def mutate_value(self):
        self.mutate_value_random()

    @staticmethod
    def generate_arg_from_signature(signature):
        if isinstance(signature, bool):
            return TFArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TFArgument(signature, ArgType.INT)
        if isinstance(signature, float):
            return TFArgument(signature, ArgType.FLOAT)
        if isinstance(signature, str):
            return TFArgument(signature, ArgType.STR)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.LIST)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.TUPLE)

        if (not isinstance(signature, dict)):
            return TFArgument(None, ArgType.NULL)

        if "type" not in signature and "Label" not in signature:
            return TFArgument(None, ArgType.NULL)

        label = signature["type"] if "type" in signature else signature["Label"]

        if label == "tf_object":
            if "class_name" not in signature:
                return TFArgument(None, ArgType.TF_OBJECT)

            if signature["class_name"] == "tensorflow.python.keras.engine.keras_tensor.KerasTensor" or \
                signature["class_name"] == "tensorflow.python.ops.variables.RefVariable":
                dtype = signature["dtype"]
                shape = signature["shape"]
                dtype = TFArgument.str_to_dtype(dtype)
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
                name = signature["to_str"].replace("<dtype: '", "").replace("'>", "")
                value = eval("tf." + name)
                return TFArgument(value, ArgType.TF_DTYPE)
            try:
                value = eval(signature.class_name)
            except:
                value = None
            return TFArgument(value, ArgType.TF_OBJECT)
        if label == "raw":
            try:
                value = json.loads(signature['value'])
            except:
                value = signature['value']
                pass
            if isinstance(value, int):
                return TFArgument(value, ArgType.INT)
            if isinstance(value, str):
                return TFArgument(value, ArgType.STR)
            if isinstance(value, float):
                return TFArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)

        if label == "tuple":
            try:
                value = json.loads(signature['value'])
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label == "list":
            try:
                try:
                    value = json.loads(signature['value'])
                except:
                    value = signature['value']
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label in ["tensor", "KerasTensor", "variable", "nparray"]:
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature["dtype"]
            dtype = TFArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            else:
                minv, maxv = 0, 1
                shape = [1, ]  
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)

        return TFArgument(None, ArgType.NULL)

class TFAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        self.record = TFDatabase.get_rand_record(api_name) if record is None else record
        self.args = TFAPI.generate_args_from_record(self.record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code_oracle(self,
                prefix="arg", oracle=OracleType.CRASH) -> str:
        
        if oracle == OracleType.CRASH:
            code = self.to_code(prefix=prefix, res_name=RESULT_KEY)
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.CUDA:
            cpu_code = self.to_code(prefix=prefix, res_name=RES_CPU_KEY, 
                use_try=True, err_name=ERR_CPU_KEY, wrap_device=True, device_name="CPU")
            gpu_code = self.to_diff_code(prefix=prefix, res_name=RES_GPU_KEY,
                use_try=True, err_name=ERR_GPU_KEY, wrap_device=True, device_name="GPU:0")
            
            code = cpu_code + gpu_code
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.PRECISION:
            low_code = self.to_code(prefix=prefix, res_name=RES_LOW_KEY, low_precision=True,
                use_try=True, err_name=ERR_LOW_KEY, time_it=True, time_var=TIME_LOW_KEY)
            high_code = self.to_diff_code(prefix=prefix, res_name=RES_HIGH_KEY,
                use_try=True, err_name=ERR_HIGH_KEY, time_it=True, time_var=TIME_HIGH_KEY)
            code = low_code + high_code
            return self.wrap_try(code, ERROR_KEY)
        return ''

    @staticmethod
    def generate_args_from_record(record: dict):

        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures['Label'] == 'list':
                    s = signatures['value']
                    if isinstance(s, list):
                        signatures = s
            args = []
            if signatures == None:
                return args
            for signature in signatures:
                x = TFArgument.generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[key] = TFArgument(value, ArgType.LIST)
            elif key != "output_signature":
                args[key] = TFArgument.generate_arg_from_signature(record[key])
        return args

    def _to_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str
        
    def _to_diff_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_diff_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_diff_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str

    def to_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            arg_code += f"{cls_name} = {self.api}({arg_str})\n"
            if inputs:
                arg_code += inputs.to_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def to_diff_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_diff_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            res_code = f""
            if inputs:
                arg_code += inputs.to_diff_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def _to_res_code(self, res_name, arg_str, input_name=None, prefix="arg"):
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            if input_name:
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        return res_code

    def _to_invocation_code(self, arg_code, res_code, use_try=False, err_name="", 
        wrap_device=False, device_name="", time_it=False, time_var="", **kwargs) -> str:
        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    @staticmethod
    def wrap_try(code:str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f"except Exception as e:\n  {RES_KEY}[\"{error_var}\"] = \"Error:\"+str(e)\n"
        return wrapped_code

    @staticmethod
    def wrap_device(code:str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code:str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f"{RES_KEY}[\"{time_var}\"] = t_end - t_start\n"
        return wrapped_code


        
def test_tf_arg():
    arg = TFArgument(None, ArgType.TF_TENSOR, shape=[2, 2], dtype=tf.int64)
    arg.mutate_value()
    print(arg.to_code("var"))
    print(arg.to_code("var", True))

def test_tf_api():
    api_name = "tf.keras.layers.Conv2D"
    record = TFDatabase.get_rand_record(api_name)
    api = TFAPI(api_name, record)
    api.mutate()
    print(api.to_code_oracle(oracle=OracleType.CRASH))
    print(api.to_code_oracle(oracle=OracleType.CUDA))
    print(api.to_code_oracle(oracle=OracleType.PRECISION))

if __name__ == '__main__':
    # test_tf_arg()
    test_tf_api()


================================================================================
FILE: src\classes\tf_library.py
================================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import tensorflow as tf
import time
import numpy as np

from classes.argument import Argument, ArgType
from classes.tf_api import TFAPI, TFArgument
from classes.library import Library
from classes.database import TFDatabase
from constants.enum import OracleType
from constants.keys import ERR_CPU_KEY, ERR_GPU_KEY, ERR_HIGH_KEY, ERR_LOW_KEY, ERROR_KEY, RES_CPU_KEY, RES_GPU_KEY, TIME_HIGH_KEY, TIME_LOW_KEY

class TFLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold
    
    def test_with_oracle(self, api: TFAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import tensorflow as tf\n"
            code += self.generate_code(api, oracle)

            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            results, error = self.run_code(code)
            if error == None:
                self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
        elif oracle == OracleType.CUDA:
            code = "import tensorflow as tf\n"
            code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            err_cpu = results[ERR_CPU_KEY]
            err_gpu = results[ERR_GPU_KEY]
            write_dir = ""
            if error is None:
                if (err_cpu is None) != (err_gpu is None):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif err_cpu == None:
                    res_cpu = results[RES_CPU_KEY]
                    res_gpu = results[RES_GPU_KEY]
                    if self.is_equal(res_cpu, res_gpu):
                        write_dir = join(self.output[oracle], "success")
                    else:
                        write_dir = join(self.output[oracle], "potential-bug")
                elif "SystemError" in err_cpu or "SystemError" in err_gpu:
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "SystemError" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import tensorflow as tf\n"
            code += "import time\n"
            code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            err_high = results[ERR_HIGH_KEY]
            err_low = results[ERR_LOW_KEY]
            write_dir = ""
            if error is None:
                if (err_high is None) != (err_low is None):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif err_high == None:
                    time_high = results[TIME_HIGH_KEY]
                    time_low = results[TIME_LOW_KEY]
                    if time_low >= self.time_bound * time_high and time_high >= self.time_thresold:
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                elif "SystemError" in err_high or "SystemError" in err_low:
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "SystemError" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: TFAPI, oracle: OracleType) -> str:
        code = ""
        if oracle == OracleType.CRASH:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.CUDA:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.PRECISION:
            code += api.to_code_oracle(oracle=oracle)
            return code
        else:
            assert(0)
    
    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_HIGH_KEY] = None
        results[ERR_LOW_KEY] = None
        
        exec(code)
        error = results[ERROR_KEY] if ERROR_KEY in results else None
        return results, error
    
    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, tf.Tensor):
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        else:
            return ArgType.TF_OBJECT

    
    @staticmethod
    def _eval_k(x):
        return tf.convert_to_tensor(x).numpy()

    @staticmethod
    def get_tensor_value(t):
        if isinstance(t, tf.SparseTensor):
            return tf.sparse.to_dense(t).numpy()
        else:
            return t.numpy()
            
    @staticmethod
    def is_equal(x, y):
        x_type = TFArgument.get_type(x)
        y_type = TFArgument.get_type(y)
        if x_type != y_type:
            return False
        if x_type == ArgType.KERAS_TENSOR:
            return tf.math.equal(x, y)
        if x_type == ArgType.TF_TENSOR:
            try:
                if isinstance(x, tf.RaggedTensor) != isinstance(y, tf.RaggedTensor):
                    return False
                if isinstance(x, tf.RaggedTensor):
                    s = tf.math.equal(x, y)
                    return s.flat_values.numpy().all()
                np_x = TFLibrary.get_tensor_value(x)
                np_y = TFLibrary.get_tensor_value(y)
                if x.dtype.is_floating:
                    return tf.experimental.numpy.allclose(np_x, np_y, rtol=1e-3, atol=1e-4)
                elif x.dtype.is_integer:
                    return np.equal(np_x, np_y).all()
            except:
                raise ValueError(f"Comparison between {type(x)} is not supported now.")
            return True
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < 1e-5
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TFLibrary.is_equal(x[i], y[i]) == False:
                    return False
            return True
        
        else:
            try:
                flag = x == y
            except:
                return True

            if isinstance(flag, np.ndarray):
                flag = flag.all()
            try:
                if flag:
                    pass
            except:
                flag = True
            return flag
    


================================================================================
FILE: src\classes\torch_api.py
================================================================================
import torch
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
from os.path import join

class TorchArgument(Argument):
    _supported_types = [
        ArgType.TORCH_DTYPE, ArgType.TORCH_OBJECT, ArgType.TORCH_TENSOR
    ]
    _dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.complex64, torch.complex128, torch.bool
    ]
    _memory_format = [
        torch.contiguous_format, torch.channels_last, torch.preserve_format
    ]

    def __init__(self,
                 value,
                 type: ArgType,
                 shape=None,
                 dtype=None,
                 max_value=1,
                 min_value=0):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value

    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision,
                                              is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            if is_cuda:
                suffix = ".cuda()"
            if dtype.is_floating_point:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype.is_complex:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype == torch.bool:
                code = f"{var_name}_tensor = torch.randint(0,2,{self.shape}, dtype={dtype})\n"
            else:
                code = f"{var_name}_tensor = torch.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert (0)

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TORCH_TENSOR:
                self.shape = [2, 2]
                self.dtype = torch.float32
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:
                self.value = choice(self._memory_format)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert (0)

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            max_value = 2
            min_value = 0
        elif dtype == torch.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == torch.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == torch.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":
            return TorchArgument(None,
                                 ArgType.TORCH_TENSOR,
                                 shape=[2, 2],
                                 dtype=torch.float32)
        if signature == "torchdtype":
            return TorchArgument(choice(TorchArgument._dtypes),
                                 ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_format)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            assert(0)
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature['dtype']
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = TorchArgument.random_tensor_value(dtype)
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape,
                                     dtype=dtype,
                                     max_value=max_value,
                                     min_value=min_value)
            else:
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape=[2, 2],
                                     dtype=torch.float32)
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT


class TorchAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)
        if record == None:
            record = TorchDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TorchDatabase.select_rand_over_db(
                    self.api, arg_name)
                if success:
                    new_arg = TorchArgument.generate_arg_from_signature(
                        new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code(self,
                prefix="arg",
                res="res",
                is_cuda=False,
                use_try=False,
                error_res=None,
                low_precision=False) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision,
                                is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           low_precision)

    def to_diff_code(self,
                     oracle: OracleType,
                     prefix="arg",
                     res="res",
                     *,
                     error_res=None,
                     use_try=False) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(arg_name, oracle)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           oracle == OracleType.PRECISION)

    @staticmethod
    def invocation_code(res, error_res, res_code, use_try, low_precision):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n  {error_res} = \"ERROR:\"+str(e)\n"
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = TorchArgument.generate_arg_from_signature(
                    record[key])
        return args

================================================================================
FILE: src\classes\torch_library.py
================================================================================
from classes.torch_api import *
from classes.library import Library
from classes.argument import *
from classes.api import *
from os.path import join
import os
from constants.keys import *

class TorchLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold
    
    def test_with_oracle(self, api: TorchAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import torch\n"
            code += self.generate_code(api, oracle)
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            _, error = self.run_code(code)
            if error == None:
                self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
            elif self.is_crash_msg(error):
                self.write_to_dir(join(self.output[oracle], "potential-bug"), api.api, code)
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
        elif oracle == OracleType.CUDA:
            code = "import torch\n"
            code += api.to_code(res=f"{RES_KEY}[\"{RES_CPU_KEY}\"]", use_try=True, error_res=f"{RES_KEY}[\"{ERR_CPU_KEY}\"]")
            code += api.to_diff_code(oracle, res=f"{RES_KEY}[\"{RES_GPU_KEY}\"]", use_try=True, error_res=f"{RES_KEY}[\"{ERR_GPU_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)

            write_dir = ""
            if error == None:
                # first check the correctness
                if results[ERR_CPU_KEY] == None and results[ERR_GPU_KEY] == None:
                    try:
                        is_equal = self.is_equal(results[RES_CPU_KEY], results[RES_GPU_KEY], self.diff_bound)
                    except Exception:
                        write_dir = join(self.output[oracle], "compare-bug")
                    else:
                        if is_equal:
                            write_dir = join(self.output[oracle], "success")
                        else:
                            write_dir = join(self.output[oracle], "potential-bug")
                elif self.is_crash_msg(results[ERR_CPU_KEY]) or self.is_crash_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif results[ERR_CPU_KEY] and results[ERR_GPU_KEY]:
                    write_dir = join(self.output[oracle], "success")
                    pass
                elif self.is_error_msg(results[ERR_CPU_KEY]) != self.is_error_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif self.is_crash_msg(error):
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import torch\n"
            code += "import time\n"
            code += api.to_code(res=f"results[\"{TIME_LOW_KEY}\"]", low_precision=True)
            code += api.to_diff_code(oracle, res=f"results[\"{TIME_HIGH_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            if error == None:
                if isinstance(results[TIME_LOW_KEY], float) and isinstance(results[TIME_HIGH_KEY], float):
                    if results[TIME_LOW_KEY] > self.time_bound * results[TIME_HIGH_KEY] and results[TIME_HIGH_KEY] > self.time_thresold:
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                else:
                    write_dir = join(self.output[oracle], "fail")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: TorchAPI, oracle: OracleType) -> str:
        if oracle == OracleType.CRASH:
            return api.to_code()
        elif oracle == OracleType.CUDA:
            code = api.to_code(res="cpu_res", use_try=True)
            code += api.to_diff_code(oracle, res="cuda_res", use_try=True)
            return code
        elif oracle == OracleType.PRECISION:
            code = api.to_code(res="low_res", low_precision=True)
            code += api.to_diff_code(oracle, res="high_res")
            return code
        else:
            assert(0)
    
    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error
    
    @staticmethod
    def is_equal(x, y, diff_bound):
        def eq_float_tensor(x, y):
            # not strictly equal
            return torch.allclose(x, y, atol=diff_bound, equal_nan=True)

        x_type = TorchArgument.get_type(x)
        y_type = TorchArgument.get_type(y)
        if x_type != y_type:
            if x_type == ArgType.TORCH_TENSOR and y_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in y:
                    flag = flag or TorchLibrary.is_equal(x, temp, diff_bound)
                return flag
            elif y_type == ArgType.TORCH_TENSOR and x_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in x:
                    flag = flag or TorchLibrary.is_equal(y, temp, diff_bound)
                return flag
            return False
        if x_type == ArgType.TORCH_TENSOR:
            x = x.cpu()
            y = y.cpu()
            if x.dtype != y.dtype or x.shape != y.shape:
                return False
            if x.is_sparse:
                x = x.to_dense()
            if y.is_sparse:
                y = y.to_dense()
            if x.is_complex():
                if not y.is_complex(): return False
                return eq_float_tensor(x.real, y.real) and eq_float_tensor(
                    x.imag, y.imag)
            if not x.dtype.is_floating_point:
                return torch.equal(x.cpu(), y.cpu())
            return eq_float_tensor(x, y)
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < diff_bound
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TorchLibrary.is_equal(x[i], y[i], diff_bound) == False:
                    return False
            return True
        else:
            return x == y
    
    @staticmethod
    def is_error_msg(error_msg):
        allowed_msgs = ["not implement", "not support"]

        if error_msg == None:
            return False
        for msg in allowed_msgs:
            if msg in error_msg:
                return False
        return True
    
    @staticmethod
    def is_crash_msg(error_msg):
        if error_msg == None:
            return False
        if "INTERNAL ASSERT" in error_msg:
            return True
        else:
            return False
    

def test():
    api_name = "torch.nn.Conv2d"
    api = TorchAPI(api_name)
    MyPytorch = TorchLibrary("torch-output")
    print(MyPytorch.generate_code(api, OracleType.CRASH))
    print(MyPytorch.generate_code(api, OracleType.CUDA))
    print(MyPytorch.generate_code(api, OracleType.PRECISION))
    MyPytorch.test_with_oracle(api, OracleType.CRASH)
    MyPytorch.test_with_oracle(api, OracleType.CUDA)
    MyPytorch.test_with_oracle(api, OracleType.PRECISION)
    # print(TorchArgument.get_type(1))

================================================================================
FILE: src\config\skip_torch.txt
================================================================================
torch.nn.AdaptiveAvgPool2d
torch.nn.AdaptiveAvgPool3d
torch.nn.functional.adaptive_avg_pool2d
torch.nn.functional.adaptive_avg_pool3d
torch.save
torch.lu_solve
torch.combinations
torch.nn.RReLU
torch.nn.functional.rrelu

================================================================================
FILE: src\constants\enum.py
================================================================================

from enum import IntEnum
class OracleType(IntEnum):
    CRASH = 1
    CUDA = 2
    PRECISION = 3

================================================================================
FILE: src\constants\keys.py
================================================================================
# execution result
RES_KEY = "results"
RESULT_KEY = "res"
ERROR_KEY = "err"
ERR_ARG_KEY = "error_args"
ERR_CPU_KEY = "err_cpu"
ERR_GPU_KEY = "err_gpu"
RES_CPU_KEY = "res_cpu"
RES_GPU_KEY = "res_gpu"
ERR_HIGH_KEY = "err_high"
ERR_LOW_KEY = "err_low"
RES_HIGH_KEY = "res_high"
RES_LOW_KEY = "res_low"
TIME_LOW_KEY = "time_low"
TIME_HIGH_KEY = "time_high"


================================================================================
FILE: src\instrumentation\torch\decorate_cls.py
================================================================================
import json
from write_tools import write_fn

def decorate_class(klass, hint):
    if not hasattr(klass, '__call__'):
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()

    def json_serialize(v):
        try:
            json.dumps(v)
            return v
        except Exception as e:
            if hasattr(v, '__name__'):
                return v.__name__
            elif hasattr(v, '__class__'):
                res = []
                if isinstance(v, tuple) or isinstance(v, list):
                    for vi in v:
                        if hasattr(vi, 'shape'):
                            res.append(get_var_signature(vi))
                        elif isinstance(vi, tuple) or isinstance(vi, list):
                            res2 = []
                            for vii in vi:
                                if (hasattr(vii, 'shape')):
                                    res2.append(get_var_signature(vii))
                            res.append(res2)
                    return res
                else:
                    return v.__class__.__module__ + v.__class__.__name__
            return str(type(v))

    def build_param_dict(*args, **kwargs):
        param_dict = dict()
        for ind, arg in enumerate(args):
            param_dict['parameter:%d' % ind] = json_serialize(arg)
        for key, value in kwargs.items():
            param_dict[key] = json_serialize(value)
        return dict(param_dict)

    def get_var_shape(var):
        if hasattr(var, 'shape'):
            s = var.shape
            if isinstance(s, list):
                return s
            elif isinstance(s, tuple):
                return list(s)
            else:
                try:
                    return list(s)  # convert torch.Size to list
                except Exception as e:
                    print(e.message)

    def get_var_dtype(var):
        if hasattr(var, 'dtype'):
            return str(var.dtype)  # string
        if isinstance(var, list):
            res = '['
            for varx in var:
                res += type(varx).__name__ + ","
            return res[:-1] + "]"  # remove the ending ","
        elif isinstance(var, tuple):
            res = '['
            for varx in var:
                res += type(varx).__name__ + ","
            return res[:-1] + "]"
        else:
            try:
                return type(var).__name__
            except Exception as e:
                print(e.message)

    def get_shape_for_tensors(t):
        if isinstance(t, list) or isinstance(t, tuple):
            input_shape = [get_var_shape(i) for i in t]
        else:
            input_shape = get_var_shape(t)
        return input_shape

    def get_var_signature(var):
        s = dict()
        s['shape'] = get_var_shape(var)
        s['dtype'] = get_var_dtype(var)
        return s

    def get_signature_for_tensors(t):
        if isinstance(t, list) or isinstance(t, tuple):
            signatures = [get_var_signature(i) for i in t]
        else:
            signatures = get_var_signature(t)
        return signatures

    def new_init(self, *args, **kwargs):
        nonlocal init_params
        init_params = build_param_dict(*args, **kwargs)
        old_init(self, *args, **kwargs)

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params
        input_signature = get_signature_for_tensors(inputs)
        outputs = old_call(self, *inputs, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        write_fn(hint, dict(init_params), input_signature, output_signature)
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass


================================================================================
FILE: src\instrumentation\torch\decorate_func.py
================================================================================
from functools import wraps
import json
import os
from write_tools import write_fn


def decorate_function(func, hint):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def json_serialize(v):
            """Return a json serializable object. """
            try:
                json.dumps(v)
                return v  # v is a int, float, list, ...
            except Exception as e:
                if hasattr(v, 'shape'):  # v numpy array
                    return get_var_signature(
                        v)  # A dict of signature {'shape':..., 'type':...}
                if hasattr(v, '__name__'):  #  v is a function
                    return v.__name__
                elif hasattr(v, '__class__'):  # v is a class
                    res = []
                    if isinstance(v, tuple) or isinstance(v, list):
                        for vi in v:
                            if hasattr(vi, 'shape'):
                                res.append(get_var_signature(vi))
                            elif isinstance(vi, tuple) or isinstance(vi, list):
                                res2 = []
                                for vii in vi:
                                    if (hasattr(vii, 'shape')):
                                        res2.append(get_var_signature(vii))
                                res.append(res2)
                        return res
                    else:
                        return v.__class__.__module__ + v.__class__.__name__  # v.name
                else:
                    raise Exception('Error [json serialize ] %s' % v)

        def build_param_dict(*args, **kwargs):
            param_dict = dict()
            for ind, arg in enumerate(args):
                param_dict['parameter:%d' % ind] = json_serialize(arg)
            for key, value in kwargs.items():
                param_dict[key] = json_serialize(value)
            return param_dict

        def get_var_shape(var):
            if hasattr(var, 'shape'):  # var is numpy.ndarray or tensor
                s = var.shape
                if isinstance(s, list):
                    return s
                elif isinstance(s, tuple):
                    return list(s)
                else:
                    try:
                        return list(s)  # convert torch.Size to list
                    except Exception as e:
                        print(e.message)

        def get_var_dtype(var):
            if hasattr(var, 'dtype'):
                return str(var.dtype)  # string
            if isinstance(var, list):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"  # remove the ending ","
            elif isinstance(var, tuple):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"
            else:
                try:
                    return type(var).__name__
                except Exception as e:
                    print(e.message)

        def get_shape_for_tensors(t):
            if isinstance(t, list):
                input_shape = [get_var_shape(i) for i in t]
            else:
                input_shape = get_var_shape(t)
            return input_shape

        def get_var_signature(var):
            s = dict()
            s['shape'] = get_var_shape(var)
            s['dtype'] = get_var_dtype(var)
            return s

        def get_signature_for_tensors(t):
            if isinstance(t, list):
                signatures = [get_var_signature(i) for i in t]
            else:
                signatures = get_var_signature(t)
            return signatures

        outputs = func(*args, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None, output_signature)
        return outputs

    if not callable(func):
        return func

    return wrapper


================================================================================
FILE: src\instrumentation\torch\README.md
================================================================================
# Instrumentation 

This folder contains code to perform instrumentation on Pytorch in order to collect dynamic execution information.

We hook the invocation of `630` Pytorch APIs in total, and the API names are listed in `torch.*` files.

## Usage:

(1) Copy the files (except `__init__.py`) under this `instrumentation` folder to the root directory where Pytorch is installed. You may want to obtain the path by running the following commands:
```
import torch
print(torch.__path__)
```
And it should return something similar to `.../lib64/python3.6/site-packages/torch`.

(2) Append the lines from the file `__init__.py` in this directory to the end of the `__init__.py` file in the root directory of installed pytorch, which should be similar to `.../lib64/python3.6/site-packages/torch/__init__.py`

(3) Configure your MongoDB in the file `write_tools.py` and then run the code where Pytorch APIs are invoked. The traced dynamic execution information for each API invocation will be added to the MongoDB.


================================================================================
FILE: src\instrumentation\torch\torch.nn.functional.txt
================================================================================
nn.functional.conv1d
nn.functional.conv2d
nn.functional.conv3d
nn.functional.conv_transpose1d
nn.functional.conv_transpose2d
nn.functional.conv_transpose3d
nn.functional.unfold
nn.functional.fold
nn.functional.avg_pool1d
nn.functional.avg_pool2d
nn.functional.avg_pool3d
nn.functional.max_pool1d
nn.functional.max_pool2d
nn.functional.max_pool3d
nn.functional.max_unpool1d
nn.functional.max_unpool2d
nn.functional.max_unpool3d
nn.functional.lp_pool1d
nn.functional.lp_pool2d
nn.functional.adaptive_max_pool1d
nn.functional.adaptive_max_pool2d
nn.functional.adaptive_max_pool3d
nn.functional.adaptive_avg_pool1d
nn.functional.adaptive_avg_pool2d
nn.functional.adaptive_avg_pool3d
nn.functional.threshold
nn.functional.threshold_
nn.functional.relu
nn.functional.relu_
nn.functional.hardtanh
nn.functional.hardtanh_
nn.functional.hardswish
nn.functional.relu6
nn.functional.elu
nn.functional.elu_
nn.functional.selu
nn.functional.celu
nn.functional.leaky_relu
nn.functional.leaky_relu_
nn.functional.prelu
nn.functional.rrelu
nn.functional.rrelu_
nn.functional.glu
nn.functional.gelu
nn.functional.logsigmoid
nn.functional.hardshrink
nn.functional.tanhshrink
nn.functional.softsign
nn.functional.softplus
nn.functional.softmin
nn.functional.softmax
nn.functional.softshrink
nn.functional.gumbel_softmax
nn.functional.log_softmax
nn.functional.tanh
nn.functional.sigmoid
nn.functional.hardsigmoid
nn.functional.silu
nn.functional.batch_norm
nn.functional.instance_norm
nn.functional.layer_norm
nn.functional.local_response_norm
nn.functional.normalize
nn.functional.linear
nn.functional.bilinear
nn.functional.dropout
nn.functional.alpha_dropout
nn.functional.feature_alpha_dropout
nn.functional.dropout2d
nn.functional.dropout3d
nn.functional.embedding
nn.functional.embedding_bag
nn.functional.one_hot
nn.functional.pairwise_distance
nn.functional.cosine_similarity
nn.functional.pdist
nn.functional.binary_cross_entropy
nn.functional.binary_cross_entropy_with_logits
nn.functional.poisson_nll_loss
nn.functional.cosine_embedding_loss
nn.functional.cross_entropy
nn.functional.ctc_loss
nn.functional.hinge_embedding_loss
nn.functional.kl_div
nn.functional.l1_loss
nn.functional.mse_loss
nn.functional.margin_ranking_loss
nn.functional.multilabel_margin_loss
nn.functional.multilabel_soft_margin_loss
nn.functional.multi_margin_loss
nn.functional.nll_loss
nn.functional.smooth_l1_loss
nn.functional.soft_margin_loss
nn.functional.triplet_margin_loss
nn.functional.triplet_margin_with_distance_loss
nn.functional.pixel_shuffle
nn.functional.pixel_unshuffle
nn.functional.pad
nn.functional.interpolate
nn.functional.upsample
nn.functional.upsample_nearest
nn.functional.upsample_bilinear
nn.functional.grid_sample
nn.functional.affine_grid
nn.parallel.data_parallel


================================================================================
FILE: src\instrumentation\torch\torch.nn.txt
================================================================================
nn.parameter.UninitializedParameter
nn.Module
nn.ModuleList
nn.ModuleDict
nn.ParameterList
nn.ParameterDict
nn.modules.module.register_module_forward_pre_hook
nn.modules.module.register_module_forward_hook
nn.modules.module.register_module_backward_hook
nn.Conv1d
nn.Conv2d
nn.Conv3d
nn.ConvTranspose1d
nn.ConvTranspose2d
nn.ConvTranspose3d
nn.LazyConv1d
nn.LazyConv2d
nn.LazyConv3d
nn.LazyConvTranspose1d
nn.LazyConvTranspose2d
nn.LazyConvTranspose3d
nn.Unfold
nn.Fold
nn.MaxPool1d
nn.MaxPool2d
nn.MaxPool3d
nn.MaxUnpool1d
nn.MaxUnpool2d
nn.MaxUnpool3d
nn.AvgPool1d
nn.AvgPool2d
nn.AvgPool3d
nn.FractionalMaxPool2d
nn.LPPool1d
nn.LPPool2d
nn.AdaptiveMaxPool1d
nn.AdaptiveMaxPool2d
nn.AdaptiveMaxPool3d
nn.AdaptiveAvgPool1d
nn.AdaptiveAvgPool2d
nn.AdaptiveAvgPool3d
nn.ReflectionPad1d
nn.ReflectionPad2d
nn.ReplicationPad1d
nn.ReplicationPad2d
nn.ReplicationPad3d
nn.ZeroPad2d
nn.ConstantPad1d
nn.ConstantPad2d
nn.ConstantPad3d
nn.ELU
nn.Hardshrink
nn.Hardsigmoid
nn.Hardtanh
nn.Hardswish
nn.LeakyReLU
nn.LogSigmoid
nn.MultiheadAttention
nn.PReLU
nn.ReLU
nn.ReLU6
nn.RReLU
nn.SELU
nn.CELU
nn.GELU
nn.Sigmoid
nn.SiLU
nn.Softplus
nn.Softshrink
nn.Softsign
nn.Tanh
nn.Tanhshrink
nn.Threshold
nn.Softmin
nn.Softmax
nn.Softmax2d
nn.LogSoftmax
nn.AdaptiveLogSoftmaxWithLoss
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
nn.GroupNorm
nn.SyncBatchNorm
nn.InstanceNorm1d
nn.InstanceNorm2d
nn.InstanceNorm3d
nn.LayerNorm
nn.LocalResponseNorm
nn.RNNBase
nn.RNN
nn.LSTM
nn.GRU
nn.RNNCell
nn.LSTMCell
nn.GRUCell
nn.Transformer
nn.TransformerEncoder
nn.TransformerDecoder
nn.TransformerEncoderLayer
nn.TransformerDecoderLayer
nn.Identity
nn.Linear
nn.Bilinear
nn.LazyLinear
nn.Dropout
nn.Dropout2d
nn.Dropout3d
nn.AlphaDropout
nn.Embedding
nn.EmbeddingBag
nn.CosineSimilarity
nn.PairwiseDistance
nn.L1Loss
nn.MSELoss
nn.CrossEntropyLoss
nn.NLLLoss
nn.CTCLoss
nn.PoissonNLLLoss
nn.GaussianNLLLoss
nn.KLDivLoss
nn.BCELoss
nn.BCEWithLogitsLoss
nn.MarginRankingLoss
nn.HingeEmbeddingLoss
nn.MultiLabelMarginLoss
nn.SmoothL1Loss
nn.SoftMarginLoss
nn.MultiLabelSoftMarginLoss
nn.CosineEmbeddingLoss
nn.MultiMarginLoss
nn.TripletMarginLoss
nn.TripletMarginWithDistanceLoss
nn.PixelShuffle
nn.PixelUnshuffle
nn.Upsample
nn.UpsamplingNearest2d
nn.UpsamplingBilinear2d
nn.ChannelShuffle
nn.DataParallel
nn.parallel.DistributedDataParallel
nn.utils.clip_grad_norm_
nn.utils.clip_grad_value_
nn.utils.parameters_to_vector
nn.utils.vector_to_parameters
nn.utils.prune.BasePruningMethod
nn.utils.prune.PruningContainer
nn.utils.prune.Identity
nn.utils.prune.RandomUnstructured
nn.utils.prune.L1Unstructured
nn.utils.prune.RandomStructured
nn.utils.prune.LnStructured
nn.utils.prune.CustomFromMask
nn.utils.prune.identity
nn.utils.prune.random_unstructured
nn.utils.prune.l1_unstructured
nn.utils.prune.random_structured
nn.utils.prune.ln_structured
nn.utils.prune.global_unstructured
nn.utils.prune.custom_from_mask
nn.utils.prune.remove
nn.utils.prune.is_pruned
nn.utils.weight_norm
nn.utils.remove_weight_norm
nn.utils.spectral_norm
nn.utils.remove_spectral_norm
nn.utils.rnn.PackedSequence
nn.utils.rnn.pack_padded_sequence
nn.utils.rnn.pad_packed_sequence
nn.utils.rnn.pad_sequence
nn.utils.rnn.pack_sequence
nn.Flatten
nn.Unflatten
nn.modules.lazy.LazyModuleMixin


================================================================================
FILE: src\instrumentation\torch\torch.txt
================================================================================
is_tensor
is_storage
is_complex
is_floating_point
is_nonzero
set_default_dtype
get_default_dtype
set_default_tensor_type
numel
set_printoptions
set_flush_denormal
rand
rand_like
randn
randn_like
randint
randint_like
randperm
empty
tensor
sparse_coo_tensor
as_tensor
as_strided
from_numpy
zeros
zeros_like
ones
ones_like
arange
range
linspace
logspace
eye
empty_like
empty_strided
full
full_like
quantize_per_tensor
quantize_per_channel
dequantize
complex
imag
polar
angle
heaviside
cat
chunk
column_stack
dstack
gather
hstack
index_select
masked_select
movedim
moveaxis
narrow
nonzero
reshape
row_stack
vstack
scatter
scatter_add
split
squeeze
stack
swapaxes
transpose
swapdims
t
take
tensor_split
tile
unbind
unsqueeze
where
Generator
seed
manual_seed
initial_seed
get_rng_state
set_rng_state
bernoulli
multinomial
normal
poisson
quasirandom.SobolEngine
save
load
get_num_threads
set_num_threads
get_num_interop_threads
set_num_interop_threads
enable_grad
no_grad
abs
absolute
acos
arccos
acosh
arccosh
add
addcdiv
addcmul
asin
arcsin
asinh
arcsinh
atan
arctan
atanh
arctanh
atan2
bitwise_not
bitwise_and
bitwise_or
bitwise_xor
ceil
clamp
max
clip
conj
copysign
cos
cosh
deg2rad
div
divide
digamma
erf
erfc
erfinv
exp
exp2
expm1
fake_quantize_per_channel_affine
fake_quantize_per_tensor_affine
fix
trunc
float_power
floor
floor_divide
fmod
frac
ldexp
lerp
lgamma
log
log10
log1p
log2
logaddexp
logaddexp2
logical_and
logical_not
logical_or
logical_xor
logit
hypot
i0
igamma
igammac
mul
multiply
mvlgamma
nan_to_num
neg
negative
nextafter
polygamma
pow
rad2deg
real
reciprocal
remainder
round
rsqrt
sigmoid
sign
sgn
signbit
sin
sinc
sinh
sqrt
square
sub
subtract
tan
tanh
true_divide
xlogy
argmax
argmin
amax
amin
all
any
min
dist
logsumexp
mean
median
nanmedian
mode
norm
nansum
prod
quantile
nanquantile
std
std_mean
sum
unique
unique_consecutive
var
var_mean
count_nonzero
allclose
argsort
eq
equal
ge
greater_equal
gt
greater
isclose
isfinite
isinf
isposinf
isneginf
isnan
isreal
kthvalue
le
less_equal
lt
less
maximum
minimum
fmax
fmin
ne
not_equal
sort
topk
msort
stft
istft
bartlett_window
blackman_window
hamming_window
hann_window
kaiser_window
atleast_1d
atleast_2d
atleast_3d
bincount
block_diag
broadcast_tensors
broadcast_to
broadcast_shapes
bucketize
cartesian_prod
cdist
clone
combinations
cross
cummax
cummin
cumprod
cumsum
diag
diag_embed
diagflat
diagonal
diff
einsum
flatten
flip
fliplr
flipud
kron
rot90
gcd
histc
meshgrid
lcm
logcumsumexp
ravel
renorm
repeat_interleave
roll
searchsorted
tensordot
trace
tril
tril_indices
triu
triu_indices
vander
view_as_real
view_as_complex
addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
cholesky
cholesky_inverse
cholesky_solve
dot
eig
geqrf
ger
outer
inner
inverse
det
logdet
slogdet
lstsq
lu
lu_solve
lu_unpack
matmul
matrix_power
matrix_rank
matrix_exp
mm
mv
orgqr
ormqr
pinverse
qr
solve
svd
svd_lowrank
pca_lowrank
symeig
lobpcg
trapz
triangular_solve
vdot
compiled_with_cxx11_abi
result_type
can_cast
promote_types
use_deterministic_algorithms
are_deterministic_algorithms_enabled
_assert


================================================================================
FILE: src\instrumentation\torch\write_tools.py
================================================================================
import pymongo

"""
You should configure the database
"""
torch_db = pymongo.MongoClient(host="localhost", port=27017)["freefuzz-torch"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "torch." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    torch_db[out_fname].insert_one(params)

================================================================================
FILE: src\instrumentation\torch\__init__.py
================================================================================
import torch.nn.utils.prune

import decorate_function
import decorate_class
import inspect

def hijack(obj, func_name_str, mode=""):
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    def is_class(x):
        return inspect.isclass(x)
    def is_callable(x):
        return callable(x)

    if mode == "function":
        wrapped_func = decorate_function(orig_func, func_name_str)
    elif mode == "class":
        wrapped_func = decorate_class(orig_func, func_name_str)
    else:
        if is_class(orig_func):
            wrapped_func = decorate_class(orig_func, func_name_str)
        elif is_callable(orig_func):
            wrapped_func = decorate_function(orig_func, func_name_str)
        else:
            wrapped_func = orig_func
    setattr(module_obj, func_name, wrapped_func)


with open(__file__.replace("__init__.py", "torch.txt"), "r") as f1:
    lines = f1.readlines()
    skipped = ["enable_grad", "get_default_dtype", "load", "tensor", "no_grad", "jit"]
    for l in lines:
        l = l.strip()
        if l not in skipped:
            hijack(torch, l, mode="function")

with open(__file__.replace("__init__.py", "torch.nn.txt"), "r") as f2:
    lines = f2.readlines()
    for l in lines:
        l = l.strip()
        hijack(torch, l)

with open(__file__.replace("__init__.py", "torch.nn.functional.txt"), "r") as f3:
    lines = f3.readlines()
    for l in lines:
        l = l.strip()
        hijack(torch, l, "function")


================================================================================
FILE: src\preprocess\process_data.py
================================================================================
import pymongo
import textdistance
import re
import numpy as np
import configparser
import sys
from os.path import join

signature_collection = "signature"
similarity_collection = "similarity"

"""
Similarity Part:
This part relys on the database so I put it into this file
"""

API_def = {}
API_args = {}


def string_similar(s1, s2):
    return textdistance.levenshtein.normalized_similarity(s1, s2)


def loadAPIs(api_file='../data/torch_APIdef.txt'):
    global API_def, API_args
    with open(api_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            API_name = line.split("(")[0]
            API_args_match = re.search("\((.*)\)", line)
            try:
                API_args_text = API_args_match.group(1)
            except:
                # with open("log/tf/api_def_error.txt", 'a') as f:
                #     f.write(line + "\n")
                # continue
                raise ValueError(line)
            # print(API_args_text)
            if API_name not in API_def.keys():
                API_def[API_name] = line
                API_args[API_name] = API_args_text


def query_argname(arg_name):
    '''
    Return a list of APIs with the exact argname
    '''
    def index_name(api_name, arg_name):
        arg_names = DB[signature_collection].find_one({"api": api_name})["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    APIs = []
    for api_name in API_args.keys():
        # search from the database
        # if arg_name exists in the records of api_name, append api_name into APIs
        if api_name not in DB.list_collection_names(
        ) or arg_name not in API_args[api_name]:
            continue
        temp = DB[api_name].find_one({arg_name: {"$exists": True}})
        if temp == None:
            # since there are two forms of names for one argument, {arg_name} and parameter:{idx}
            # we need to check the parameter:{idx}
            idx_name = index_name(api_name, arg_name)
            if idx_name and DB[api_name].find_one({idx_name: {"$exists": True}}):
                APIs.append(api_name)
        else:
            APIs.append(api_name)
    return APIs


def mean_norm(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def similarAPI(API, argname):
    '''
    Return a list of similar APIs (with the same argname) and their similarities
    '''
    API_with_same_argname = query_argname(argname)
    if len(API_with_same_argname) == 0:
        return [], []
    probs = []
    original_def = API_def[API]
    for item in API_with_same_argname:
        to_compare = API_def[item]
        probs.append(string_similar(original_def, to_compare))
    prob_norm2 = softmax(probs)
    return API_with_same_argname, prob_norm2




"""
Writing Parts (Data Preprocessing):
This part is to write API signature, argument value space and similarity
to database.

You SHOULD call these functions in this order!
    1. write_API_signature
    2. write_similarity
"""

def write_API_signature(library_name='torch'):
    """
    API's signature will be stored in 'signature' collection with the form
        api: the name of api
        args: the list of arguments' names
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue
        if api_name not in API_args.keys():
            DB[signature_collection].insert_one({"api": api_name, "args": []})
            continue

        arg_names = []
        for temp_name in API_args[api_name].split(","):
            temp_name = temp_name.strip()
            if len(temp_name) == 0 or temp_name == "*":
                continue
            if "=" in temp_name:
                temp_name = temp_name[:temp_name.find("=")]
            arg_names.append(temp_name)
        DB[signature_collection].insert_one({
            "api": api_name,
            "args": arg_names
        })


def write_similarity(library_name='torch'):
    """
    Write the similarity of (api, arg) in 'similarity' with the form:
        api: the name of api
        arg: the name of arg
        APIs: the list of similar APIs
        probs: the probability list
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue

        print(api_name)
        arg_names = DB["signature"].find_one({"api": api_name})["args"]
        for arg_name in arg_names:
            APIs, probs = similarAPI(api_name, arg_name)
            sim_dict = {}
            sim_dict["api"] = api_name
            sim_dict["arg"] = arg_name
            sim_dict["APIs"] = APIs
            sim_dict["probs"] = list(probs)
            DB[similarity_collection].insert_one(sim_dict)


if __name__ == "__main__":
    target = sys.argv[1]
    if target not in ["torch", "tf"]:
        print("Only support 'torch' or 'tf'!")
        assert(0)

    """
    Database Settings
    """
    config_name = f"demo_{target}.conf"
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join("config", config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    DB = pymongo.MongoClient(host, port)[mongo_cfg[f"{target}_database"]]

    loadAPIs(join("..", "data", f"{target}_APIdef.txt"))
    write_API_signature(target)
    write_similarity(target)


================================================================================
FILE: src\utils\converter.py
================================================================================
def str_to_bool(s):
    return True if s.lower() == 'true' else False


================================================================================
FILE: src\utils\printer.py
================================================================================
def dump_data(content, file_name, mode="w"):
    with open(file_name, mode) as f:
        f.write(content)

================================================================================
FILE: src\utils\probability.py
================================================================================
from numpy.random import rand

def choose_from_list() -> bool:
    return rand() < 0.2

def change_tensor_dimension() -> bool:
    return rand() < 0.3

def add_tensor_dimension() -> bool:
    return rand() < 0.5

def change_tensor_shape() -> bool:
    return rand() < 0.3

def change_tensor_dtype() -> bool:
    return rand() < 0.3

def do_type_mutation() -> bool:
    return rand() < 0.2

def do_select_from_db() -> bool:
    return rand() < 0.2


================================================================================
FILE: src\utils\skip.py
================================================================================

with open(__file__.replace("utils/skip.py", "config/skip_torch.txt")) as f:
    skip_torch = f.read().split("\n")

with open(__file__.replace("utils/skip.py", "config/skip_tf.txt")) as f:
    skip_tf = f.read().split("\n")

def need_skip_torch(api_name):
    if api_name in skip_torch:
        return True
    else:
        return False

def need_skip_tf(api_name):
    if api_name in skip_tf:
        return True
    skip_keywords = ["tf.keras.applications", "Input", "get_file"]
    for keyword in skip_keywords:
        if keyword in api_name:
            return True
    return False
