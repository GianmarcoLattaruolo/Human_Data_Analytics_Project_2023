
фЕ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.12v2.11.1-3812-gef4eebff7d48АГ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

"Adam/v/batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_58/beta

6Adam/v/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_58/beta*
_output_shapes
:*
dtype0

"Adam/m/batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_58/beta

6Adam/m/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_58/beta*
_output_shapes
:*
dtype0

#Adam/v/batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_58/gamma

7Adam/v/batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_58/gamma*
_output_shapes
:*
dtype0

#Adam/m/batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_58/gamma

7Adam/m/batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_58/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/conv2d_transpose_28/bias

3Adam/v/conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_28/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/conv2d_transpose_28/bias

3Adam/m/conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_28/bias*
_output_shapes
:*
dtype0
Ї
!Adam/v/conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/conv2d_transpose_28/kernel
 
5Adam/v/conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/conv2d_transpose_28/kernel*'
_output_shapes
:*
dtype0
Ї
!Adam/m/conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/conv2d_transpose_28/kernel
 
5Adam/m/conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/conv2d_transpose_28/kernel*'
_output_shapes
:*
dtype0

"Adam/v/batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_57/beta

6Adam/v/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_57/beta*
_output_shapes	
:*
dtype0

"Adam/m/batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_57/beta

6Adam/m/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_57/beta*
_output_shapes	
:*
dtype0

#Adam/v/batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_57/gamma

7Adam/v/batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_57/gamma*
_output_shapes	
:*
dtype0

#Adam/m/batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_57/gamma

7Adam/m/batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_57/gamma*
_output_shapes	
:*
dtype0

Adam/v/conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/conv2d_transpose_27/bias

3Adam/v/conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_27/bias*
_output_shapes	
:*
dtype0

Adam/m/conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/conv2d_transpose_27/bias

3Adam/m/conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_27/bias*
_output_shapes	
:*
dtype0
Ї
!Adam/v/conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/conv2d_transpose_27/kernel
 
5Adam/v/conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/conv2d_transpose_27/kernel*'
_output_shapes
:*
dtype0
Ї
!Adam/m/conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/conv2d_transpose_27/kernel
 
5Adam/m/conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/conv2d_transpose_27/kernel*'
_output_shapes
:*
dtype0

"Adam/v/batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_56/beta

6Adam/v/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_56/beta*
_output_shapes
:*
dtype0

"Adam/m/batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_56/beta

6Adam/m/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_56/beta*
_output_shapes
:*
dtype0

#Adam/v/batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_56/gamma

7Adam/v/batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_56/gamma*
_output_shapes
:*
dtype0

#Adam/m/batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_56/gamma

7Adam/m/batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_56/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_38/bias
{
)Adam/v/conv2d_38/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_38/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_38/bias
{
)Adam/m/conv2d_38/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_38/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/conv2d_38/kernel

+Adam/v/conv2d_38/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_38/kernel*'
_output_shapes
:*
dtype0

Adam/m/conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/conv2d_38/kernel

+Adam/m/conv2d_38/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_38/kernel*'
_output_shapes
:*
dtype0

"Adam/v/batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_55/beta

6Adam/v/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_55/beta*
_output_shapes	
:*
dtype0

"Adam/m/batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_55/beta

6Adam/m/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_55/beta*
_output_shapes	
:*
dtype0

#Adam/v/batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_55/gamma

7Adam/v/batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_55/gamma*
_output_shapes	
:*
dtype0

#Adam/m/batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_55/gamma

7Adam/m/batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_55/gamma*
_output_shapes	
:*
dtype0

Adam/v/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_37/bias
|
)Adam/v/conv2d_37/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_37/bias*
_output_shapes	
:*
dtype0

Adam/m/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_37/bias
|
)Adam/m/conv2d_37/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_37/bias*
_output_shapes	
:*
dtype0

Adam/v/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/conv2d_37/kernel

+Adam/v/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_37/kernel*'
_output_shapes
:*
dtype0

Adam/m/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/conv2d_37/kernel

+Adam/m/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_37/kernel*'
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Є
&batch_normalization_58/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_58/moving_variance

:batch_normalization_58/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_58/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_58/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_58/moving_mean

6batch_normalization_58/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_58/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_58/beta

/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_58/beta*
_output_shapes
:*
dtype0

batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_58/gamma

0batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_58/gamma*
_output_shapes
:*
dtype0

conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_28/bias

,conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/bias*
_output_shapes
:*
dtype0

conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_28/kernel

.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*'
_output_shapes
:*
dtype0
Ѕ
&batch_normalization_57/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_57/moving_variance

:batch_normalization_57/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_57/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_57/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_57/moving_mean

6batch_normalization_57/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_57/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_57/beta

/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_57/beta*
_output_shapes	
:*
dtype0

batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_57/gamma

0batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_57/gamma*
_output_shapes	
:*
dtype0

conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_27/bias

,conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_27/kernel

.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*'
_output_shapes
:*
dtype0
Є
&batch_normalization_56/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_56/moving_variance

:batch_normalization_56/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_56/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_56/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_56/moving_mean

6batch_normalization_56/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_56/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_56/beta

/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_56/beta*
_output_shapes
:*
dtype0

batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_56/gamma

0batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_56/gamma*
_output_shapes
:*
dtype0
t
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_38/bias
m
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes
:*
dtype0

conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_38/kernel
~
$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*'
_output_shapes
:*
dtype0
Ѕ
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_55/moving_variance

:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_55/moving_mean

6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_55/beta

/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes	
:*
dtype0

batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_55/gamma

0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes	
:*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:*
dtype0

conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_37/kernel
~
$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*'
_output_shapes
:*
dtype0

serving_default_input_59Placeholder*0
_output_shapes
:џџџџџџџџџ@*
dtype0*%
shape:џџџџџџџџџ@
М
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_59conv2d_37/kernelconv2d_37/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_38/kernelconv2d_38/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_28/kernelconv2d_transpose_28/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_variance*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_8187465

NoOpNoOp
и
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bћ
Ь
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
ы
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories*

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
 layer-4
!layer-5
"layer_with_weights-2
"layer-6
#layer_with_weights-3
#layer-7
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
#*_self_saveable_object_factories*
К
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23*
z
+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15*
* 
А
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 

P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla*

Wserving_default* 
* 
* 
э
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

+kernel
,bias
#^_self_saveable_object_factories
 __jit_compiled_convolution_op*
Г
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
#f_self_saveable_object_factories* 
њ
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	-gamma
.beta
/moving_mean
0moving_variance
#n_self_saveable_object_factories*
э
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

1kernel
2bias
#u_self_saveable_object_factories
 v_jit_compiled_convolution_op*
Г
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
#}_self_saveable_object_factories* 

~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	3gamma
4beta
5moving_mean
6moving_variance
$_self_saveable_object_factories*
Z
+0
,1
-2
.3
/4
05
16
27
38
49
510
611*
<
+0
,1
-2
.3
14
25
36
47*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories* 
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$ _self_saveable_object_factories* 
ѕ
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses

7kernel
8bias
$Ї_self_saveable_object_factories
!Ј_jit_compiled_convolution_op*

Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
	Џaxis
	9gamma
:beta
;moving_mean
<moving_variance
$А_self_saveable_object_factories*
К
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
$З_self_saveable_object_factories* 
К
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses
$О_self_saveable_object_factories* 
ѕ
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses

=kernel
>bias
$Х_self_saveable_object_factories
!Ц_jit_compiled_convolution_op*

Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
	Эaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance
$Ю_self_saveable_object_factories*
Z
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11*
<
70
81
92
:3
=4
>5
?6
@7*
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
:
дtrace_0
еtrace_1
жtrace_2
зtrace_3* 
:
иtrace_0
йtrace_1
кtrace_2
лtrace_3* 
* 
PJ
VARIABLE_VALUEconv2d_37/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_37/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_55/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_55/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"batch_normalization_55/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_normalization_55/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_38/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_38/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_56/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_56/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_56/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_56/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_27/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_27/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_57/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_57/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_57/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_57/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_28/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_28/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_58/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_58/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_58/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_58/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
<
/0
01
52
63
;4
<5
A6
B7*

0
1
2*

м0
н1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ђ
Q0
о1
п2
р3
с4
т5
у6
ф7
х8
ц9
ч10
ш11
щ12
ъ13
ы14
ь15
э16
ю17
я18
№19
ё20
ђ21
ѓ22
є23
ѕ24
і25
ї26
ј27
љ28
њ29
ћ30
ќ31
§32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

о0
р1
т2
ф3
ц4
ш5
ъ6
ь7
ю8
№9
ђ10
є11
і12
ј13
њ14
ќ15*

п0
с1
у2
х3
ч4
щ5
ы6
э7
я8
ё9
ѓ10
ѕ11
ї12
љ13
ћ14
§15*
ш
ўtrace_0
џtrace_1
trace_2
trace_3
trace_4
trace_5
trace_6
trace_7
trace_8
trace_9
trace_10
trace_11
trace_12
trace_13
trace_14
trace_15* 
* 

+0
,1*

+0
,1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
 
-0
.1
/2
03*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

Ёtrace_0
Ђtrace_1* 

Ѓtrace_0
Єtrace_1* 
* 
* 

10
21*

10
21*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

Њtrace_0* 

Ћtrace_0* 
* 
* 
* 
* 
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

Бtrace_0* 

Вtrace_0* 
* 
 
30
41
52
63*

30
41*
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Иtrace_0
Йtrace_1* 

Кtrace_0
Лtrace_1* 
* 
* 
 
/0
01
52
63*
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
* 
* 
* 
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
* 

70
81*

70
81*
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
* 
* 
 
90
:1
;2
<3*

90
:1*
* 

гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 
* 
* 
* 
* 
* 

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

сtrace_0* 

тtrace_0* 
* 
* 
* 
* 

уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses* 

шtrace_0
щtrace_1* 

ъtrace_0
ыtrace_1* 
* 

=0
>1*

=0
>1*
* 

ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses*

ёtrace_0* 

ђtrace_0* 
* 
* 
 
?0
@1
A2
B3*

?0
@1*
* 

ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

јtrace_0
љtrace_1* 

њtrace_0
ћtrace_1* 
* 
* 
 
;0
<1
A2
B3*
<
0
1
2
3
 4
!5
"6
#7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ќ	variables
§	keras_api

ўtotal

џcount*
M
	variables
	keras_api

total

count

_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/conv2d_37/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_37/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_37/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_37/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_55/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/batch_normalization_55/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/batch_normalization_55/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_55/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_38/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_38/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_38/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_38/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_56/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_56/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_56/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_56/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/conv2d_transpose_27/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/conv2d_transpose_27/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/conv2d_transpose_27/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/conv2d_transpose_27/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_57/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_57/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_57/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_57/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/conv2d_transpose_28/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/conv2d_transpose_28/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/conv2d_transpose_28/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/conv2d_transpose_28/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_58/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_58/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_58/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_58/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 
* 
* 

ў0
џ1*

ќ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
г
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp0batch_normalization_56/gamma/Read/ReadVariableOp/batch_normalization_56/beta/Read/ReadVariableOp6batch_normalization_56/moving_mean/Read/ReadVariableOp:batch_normalization_56/moving_variance/Read/ReadVariableOp.conv2d_transpose_27/kernel/Read/ReadVariableOp,conv2d_transpose_27/bias/Read/ReadVariableOp0batch_normalization_57/gamma/Read/ReadVariableOp/batch_normalization_57/beta/Read/ReadVariableOp6batch_normalization_57/moving_mean/Read/ReadVariableOp:batch_normalization_57/moving_variance/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp,conv2d_transpose_28/bias/Read/ReadVariableOp0batch_normalization_58/gamma/Read/ReadVariableOp/batch_normalization_58/beta/Read/ReadVariableOp6batch_normalization_58/moving_mean/Read/ReadVariableOp:batch_normalization_58/moving_variance/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+Adam/m/conv2d_37/kernel/Read/ReadVariableOp+Adam/v/conv2d_37/kernel/Read/ReadVariableOp)Adam/m/conv2d_37/bias/Read/ReadVariableOp)Adam/v/conv2d_37/bias/Read/ReadVariableOp7Adam/m/batch_normalization_55/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_55/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_55/beta/Read/ReadVariableOp6Adam/v/batch_normalization_55/beta/Read/ReadVariableOp+Adam/m/conv2d_38/kernel/Read/ReadVariableOp+Adam/v/conv2d_38/kernel/Read/ReadVariableOp)Adam/m/conv2d_38/bias/Read/ReadVariableOp)Adam/v/conv2d_38/bias/Read/ReadVariableOp7Adam/m/batch_normalization_56/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_56/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_56/beta/Read/ReadVariableOp6Adam/v/batch_normalization_56/beta/Read/ReadVariableOp5Adam/m/conv2d_transpose_27/kernel/Read/ReadVariableOp5Adam/v/conv2d_transpose_27/kernel/Read/ReadVariableOp3Adam/m/conv2d_transpose_27/bias/Read/ReadVariableOp3Adam/v/conv2d_transpose_27/bias/Read/ReadVariableOp7Adam/m/batch_normalization_57/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_57/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_57/beta/Read/ReadVariableOp6Adam/v/batch_normalization_57/beta/Read/ReadVariableOp5Adam/m/conv2d_transpose_28/kernel/Read/ReadVariableOp5Adam/v/conv2d_transpose_28/kernel/Read/ReadVariableOp3Adam/m/conv2d_transpose_28/bias/Read/ReadVariableOp3Adam/v/conv2d_transpose_28/bias/Read/ReadVariableOp7Adam/m/batch_normalization_58/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_58/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_58/beta/Read/ReadVariableOp6Adam/v/batch_normalization_58/beta/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*K
TinD
B2@	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_8189000
і
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_37/kernelconv2d_37/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_38/kernelconv2d_38/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_28/kernelconv2d_transpose_28/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_variance	iterationlearning_rateAdam/m/conv2d_37/kernelAdam/v/conv2d_37/kernelAdam/m/conv2d_37/biasAdam/v/conv2d_37/bias#Adam/m/batch_normalization_55/gamma#Adam/v/batch_normalization_55/gamma"Adam/m/batch_normalization_55/beta"Adam/v/batch_normalization_55/betaAdam/m/conv2d_38/kernelAdam/v/conv2d_38/kernelAdam/m/conv2d_38/biasAdam/v/conv2d_38/bias#Adam/m/batch_normalization_56/gamma#Adam/v/batch_normalization_56/gamma"Adam/m/batch_normalization_56/beta"Adam/v/batch_normalization_56/beta!Adam/m/conv2d_transpose_27/kernel!Adam/v/conv2d_transpose_27/kernelAdam/m/conv2d_transpose_27/biasAdam/v/conv2d_transpose_27/bias#Adam/m/batch_normalization_57/gamma#Adam/v/batch_normalization_57/gamma"Adam/m/batch_normalization_57/beta"Adam/v/batch_normalization_57/beta!Adam/m/conv2d_transpose_28/kernel!Adam/v/conv2d_transpose_28/kernelAdam/m/conv2d_transpose_28/biasAdam/v/conv2d_transpose_28/bias#Adam/m/batch_normalization_58/gamma#Adam/v/batch_normalization_58/gamma"Adam/m/batch_normalization_58/beta"Adam/v/batch_normalization_58/betatotal_1count_1totalcount*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_8189196Фе
ц
њ*
#__inference__traced_restore_8189196
file_prefix<
!assignvariableop_conv2d_37_kernel:0
!assignvariableop_1_conv2d_37_bias:	>
/assignvariableop_2_batch_normalization_55_gamma:	=
.assignvariableop_3_batch_normalization_55_beta:	D
5assignvariableop_4_batch_normalization_55_moving_mean:	H
9assignvariableop_5_batch_normalization_55_moving_variance:	>
#assignvariableop_6_conv2d_38_kernel:/
!assignvariableop_7_conv2d_38_bias:=
/assignvariableop_8_batch_normalization_56_gamma:<
.assignvariableop_9_batch_normalization_56_beta:D
6assignvariableop_10_batch_normalization_56_moving_mean:H
:assignvariableop_11_batch_normalization_56_moving_variance:I
.assignvariableop_12_conv2d_transpose_27_kernel:;
,assignvariableop_13_conv2d_transpose_27_bias:	?
0assignvariableop_14_batch_normalization_57_gamma:	>
/assignvariableop_15_batch_normalization_57_beta:	E
6assignvariableop_16_batch_normalization_57_moving_mean:	I
:assignvariableop_17_batch_normalization_57_moving_variance:	I
.assignvariableop_18_conv2d_transpose_28_kernel::
,assignvariableop_19_conv2d_transpose_28_bias:>
0assignvariableop_20_batch_normalization_58_gamma:=
/assignvariableop_21_batch_normalization_58_beta:D
6assignvariableop_22_batch_normalization_58_moving_mean:H
:assignvariableop_23_batch_normalization_58_moving_variance:'
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: F
+assignvariableop_26_adam_m_conv2d_37_kernel:F
+assignvariableop_27_adam_v_conv2d_37_kernel:8
)assignvariableop_28_adam_m_conv2d_37_bias:	8
)assignvariableop_29_adam_v_conv2d_37_bias:	F
7assignvariableop_30_adam_m_batch_normalization_55_gamma:	F
7assignvariableop_31_adam_v_batch_normalization_55_gamma:	E
6assignvariableop_32_adam_m_batch_normalization_55_beta:	E
6assignvariableop_33_adam_v_batch_normalization_55_beta:	F
+assignvariableop_34_adam_m_conv2d_38_kernel:F
+assignvariableop_35_adam_v_conv2d_38_kernel:7
)assignvariableop_36_adam_m_conv2d_38_bias:7
)assignvariableop_37_adam_v_conv2d_38_bias:E
7assignvariableop_38_adam_m_batch_normalization_56_gamma:E
7assignvariableop_39_adam_v_batch_normalization_56_gamma:D
6assignvariableop_40_adam_m_batch_normalization_56_beta:D
6assignvariableop_41_adam_v_batch_normalization_56_beta:P
5assignvariableop_42_adam_m_conv2d_transpose_27_kernel:P
5assignvariableop_43_adam_v_conv2d_transpose_27_kernel:B
3assignvariableop_44_adam_m_conv2d_transpose_27_bias:	B
3assignvariableop_45_adam_v_conv2d_transpose_27_bias:	F
7assignvariableop_46_adam_m_batch_normalization_57_gamma:	F
7assignvariableop_47_adam_v_batch_normalization_57_gamma:	E
6assignvariableop_48_adam_m_batch_normalization_57_beta:	E
6assignvariableop_49_adam_v_batch_normalization_57_beta:	P
5assignvariableop_50_adam_m_conv2d_transpose_28_kernel:P
5assignvariableop_51_adam_v_conv2d_transpose_28_kernel:A
3assignvariableop_52_adam_m_conv2d_transpose_28_bias:A
3assignvariableop_53_adam_v_conv2d_transpose_28_bias:E
7assignvariableop_54_adam_m_batch_normalization_58_gamma:E
7assignvariableop_55_adam_v_batch_normalization_58_gamma:D
6assignvariableop_56_adam_m_batch_normalization_58_beta:D
6assignvariableop_57_adam_v_batch_normalization_58_beta:%
assignvariableop_58_total_1: %
assignvariableop_59_count_1: #
assignvariableop_60_total: #
assignvariableop_61_count: 
identity_63ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*З
value­BЊ?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHё
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B м
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesџ
ќ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_37_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_37_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_55_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_55_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_55_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_55_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_38_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_38_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_56_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_56_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_56_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_56_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_27_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_13AssignVariableOp,assignvariableop_13_conv2d_transpose_27_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_57_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_57_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_57_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_57_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_18AssignVariableOp.assignvariableop_18_conv2d_transpose_28_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_19AssignVariableOp,assignvariableop_19_conv2d_transpose_28_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_58_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_58_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_58_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_58_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_24AssignVariableOpassignvariableop_24_iterationIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_25AssignVariableOp!assignvariableop_25_learning_rateIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_m_conv2d_37_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_v_conv2d_37_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_conv2d_37_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_conv2d_37_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_30AssignVariableOp7assignvariableop_30_adam_m_batch_normalization_55_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_v_batch_normalization_55_gammaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_m_batch_normalization_55_betaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_v_batch_normalization_55_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_m_conv2d_38_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_v_conv2d_38_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_m_conv2d_38_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_v_conv2d_38_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_m_batch_normalization_56_gammaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_v_batch_normalization_56_gammaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_m_batch_normalization_56_betaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_v_batch_normalization_56_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_m_conv2d_transpose_27_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adam_v_conv2d_transpose_27_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_m_conv2d_transpose_27_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_v_conv2d_transpose_27_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_m_batch_normalization_57_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_v_batch_normalization_57_gammaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_m_batch_normalization_57_betaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_v_batch_normalization_57_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_m_conv2d_transpose_28_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_v_conv2d_transpose_28_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_m_conv2d_transpose_28_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_53AssignVariableOp3assignvariableop_53_adam_v_conv2d_transpose_28_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_m_batch_normalization_58_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_v_batch_normalization_58_gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_m_batch_normalization_58_betaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_v_batch_normalization_58_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_1Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_1Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_60AssignVariableOpassignvariableop_60_totalIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_61AssignVariableOpassignvariableop_61_countIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ѓ
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_63IdentityIdentity_62:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_63Identity_63:output:0*
_input_shapes
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
н"
Ђ
D__inference_Encoder_layer_call_and_return_conditional_losses_8186380
input_57,
conv2d_37_8186349: 
conv2d_37_8186351:	-
batch_normalization_55_8186355:	-
batch_normalization_55_8186357:	-
batch_normalization_55_8186359:	-
batch_normalization_55_8186361:	,
conv2d_38_8186364:
conv2d_38_8186366:,
batch_normalization_56_8186370:,
batch_normalization_56_8186372:,
batch_normalization_56_8186374:,
batch_normalization_56_8186376:
identityЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ!conv2d_37/StatefulPartitionedCallЂ!conv2d_38/StatefulPartitionedCall
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinput_57conv2d_37_8186349conv2d_37_8186351*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100љ
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_55_8186355batch_normalization_55_8186357batch_normalization_55_8186359batch_normalization_55_8186361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185995Г
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0conv2d_38_8186364conv2d_38_8186366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127ј
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_56_8186370batch_normalization_56_8186372batch_normalization_56_8186374batch_normalization_56_8186376*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186071
IdentityIdentity7batch_normalization_56/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ№
NoOpNoOp/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_57
Р
b
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188510

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               u
PadPadinputsPad/paddings:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџn
IdentityIdentityPad:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188361

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188791

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187877
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
§*

P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8188569

inputsC
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :U
subSubstrided_slice_1:output:0sub/y:output:0*
T0*
_output_shapes
: G
mul/yConst*
_output_shapes
: *
dtype0*
value	B :D
mulMulsub:z:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : H
sub_1Subadd:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	sub_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_2AddV2	mul_1:z:0add_2/y:output:0*
T0*
_output_shapes
: I
sub_3/yConst*
_output_shapes
: *
dtype0*
value	B : J
sub_3Sub	add_2:z:0sub_3/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_3AddV2	sub_3:z:0add_3/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :{
stackPackstrided_slice:output:0	add_1:z:0	add_3:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0о
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_8187907
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
З
Ќ
)__inference_Decoder_layer_call_fn_8188090

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186722
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
дx

 __inference__traced_save_8189000
file_prefix/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop;
7savev2_batch_normalization_56_gamma_read_readvariableop:
6savev2_batch_normalization_56_beta_read_readvariableopA
=savev2_batch_normalization_56_moving_mean_read_readvariableopE
Asavev2_batch_normalization_56_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_27_kernel_read_readvariableop7
3savev2_conv2d_transpose_27_bias_read_readvariableop;
7savev2_batch_normalization_57_gamma_read_readvariableop:
6savev2_batch_normalization_57_beta_read_readvariableopA
=savev2_batch_normalization_57_moving_mean_read_readvariableopE
Asavev2_batch_normalization_57_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop7
3savev2_conv2d_transpose_28_bias_read_readvariableop;
7savev2_batch_normalization_58_gamma_read_readvariableop:
6savev2_batch_normalization_58_beta_read_readvariableopA
=savev2_batch_normalization_58_moving_mean_read_readvariableopE
Asavev2_batch_normalization_58_moving_variance_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_adam_m_conv2d_37_kernel_read_readvariableop6
2savev2_adam_v_conv2d_37_kernel_read_readvariableop4
0savev2_adam_m_conv2d_37_bias_read_readvariableop4
0savev2_adam_v_conv2d_37_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_55_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_55_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_55_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_55_beta_read_readvariableop6
2savev2_adam_m_conv2d_38_kernel_read_readvariableop6
2savev2_adam_v_conv2d_38_kernel_read_readvariableop4
0savev2_adam_m_conv2d_38_bias_read_readvariableop4
0savev2_adam_v_conv2d_38_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_56_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_56_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_56_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_56_beta_read_readvariableop@
<savev2_adam_m_conv2d_transpose_27_kernel_read_readvariableop@
<savev2_adam_v_conv2d_transpose_27_kernel_read_readvariableop>
:savev2_adam_m_conv2d_transpose_27_bias_read_readvariableop>
:savev2_adam_v_conv2d_transpose_27_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_57_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_57_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_57_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_57_beta_read_readvariableop@
<savev2_adam_m_conv2d_transpose_28_kernel_read_readvariableop@
<savev2_adam_v_conv2d_transpose_28_kernel_read_readvariableop>
:savev2_adam_m_conv2d_transpose_28_bias_read_readvariableop>
:savev2_adam_v_conv2d_transpose_28_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_58_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_58_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_58_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_58_beta_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*З
value­BЊ?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B О
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop7savev2_batch_normalization_56_gamma_read_readvariableop6savev2_batch_normalization_56_beta_read_readvariableop=savev2_batch_normalization_56_moving_mean_read_readvariableopAsavev2_batch_normalization_56_moving_variance_read_readvariableop5savev2_conv2d_transpose_27_kernel_read_readvariableop3savev2_conv2d_transpose_27_bias_read_readvariableop7savev2_batch_normalization_57_gamma_read_readvariableop6savev2_batch_normalization_57_beta_read_readvariableop=savev2_batch_normalization_57_moving_mean_read_readvariableopAsavev2_batch_normalization_57_moving_variance_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop3savev2_conv2d_transpose_28_bias_read_readvariableop7savev2_batch_normalization_58_gamma_read_readvariableop6savev2_batch_normalization_58_beta_read_readvariableop=savev2_batch_normalization_58_moving_mean_read_readvariableopAsavev2_batch_normalization_58_moving_variance_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_adam_m_conv2d_37_kernel_read_readvariableop2savev2_adam_v_conv2d_37_kernel_read_readvariableop0savev2_adam_m_conv2d_37_bias_read_readvariableop0savev2_adam_v_conv2d_37_bias_read_readvariableop>savev2_adam_m_batch_normalization_55_gamma_read_readvariableop>savev2_adam_v_batch_normalization_55_gamma_read_readvariableop=savev2_adam_m_batch_normalization_55_beta_read_readvariableop=savev2_adam_v_batch_normalization_55_beta_read_readvariableop2savev2_adam_m_conv2d_38_kernel_read_readvariableop2savev2_adam_v_conv2d_38_kernel_read_readvariableop0savev2_adam_m_conv2d_38_bias_read_readvariableop0savev2_adam_v_conv2d_38_bias_read_readvariableop>savev2_adam_m_batch_normalization_56_gamma_read_readvariableop>savev2_adam_v_batch_normalization_56_gamma_read_readvariableop=savev2_adam_m_batch_normalization_56_beta_read_readvariableop=savev2_adam_v_batch_normalization_56_beta_read_readvariableop<savev2_adam_m_conv2d_transpose_27_kernel_read_readvariableop<savev2_adam_v_conv2d_transpose_27_kernel_read_readvariableop:savev2_adam_m_conv2d_transpose_27_bias_read_readvariableop:savev2_adam_v_conv2d_transpose_27_bias_read_readvariableop>savev2_adam_m_batch_normalization_57_gamma_read_readvariableop>savev2_adam_v_batch_normalization_57_gamma_read_readvariableop=savev2_adam_m_batch_normalization_57_beta_read_readvariableop=savev2_adam_v_batch_normalization_57_beta_read_readvariableop<savev2_adam_m_conv2d_transpose_28_kernel_read_readvariableop<savev2_adam_v_conv2d_transpose_28_kernel_read_readvariableop:savev2_adam_m_conv2d_transpose_28_bias_read_readvariableop:savev2_adam_v_conv2d_transpose_28_bias_read_readvariableop>savev2_adam_m_batch_normalization_58_gamma_read_readvariableop>savev2_adam_v_batch_normalization_58_gamma_read_readvariableop=savev2_adam_m_batch_normalization_58_beta_read_readvariableop=savev2_adam_v_batch_normalization_58_beta_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *M
dtypesC
A2?	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ї
_input_shapes
: ::::::::::::::::::::::::: : ::::::::::::::::::::::::::::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::-#)
'
_output_shapes
::-$)
'
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::-+)
'
_output_shapes
::-,)
'
_output_shapes
::!-

_output_shapes	
::!.

_output_shapes	
::!/

_output_shapes	
::!0

_output_shapes	
::!1

_output_shapes	
::!2

_output_shapes	
::-3)
'
_output_shapes
::-4)
'
_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
::;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: 
Г
с
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187196

inputs*
encoder_8187145:
encoder_8187147:	
encoder_8187149:	
encoder_8187151:	
encoder_8187153:	
encoder_8187155:	*
encoder_8187157:
encoder_8187159:
encoder_8187161:
encoder_8187163:
encoder_8187165:
encoder_8187167:*
decoder_8187170:
decoder_8187172:	
decoder_8187174:	
decoder_8187176:	
decoder_8187178:	
decoder_8187180:	*
decoder_8187182:
decoder_8187184:
decoder_8187186:
decoder_8187188:
decoder_8187190:
decoder_8187192:
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallД
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_8187145encoder_8187147encoder_8187149encoder_8187151encoder_8187153encoder_8187155encoder_8187157encoder_8187159encoder_8187161encoder_8187163encoder_8187165encoder_8187167*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186256ш
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_8187170decoder_8187172decoder_8187174decoder_8187176decoder_8187178decoder_8187180decoder_8187182decoder_8187184decoder_8187186decoder_8187188decoder_8187190decoder_8187192*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186850
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д
Y
$__inference__update_step_xla_8187872
gradient#
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:: *
	_noinline(:Q M
'
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
с"
Ђ
D__inference_Encoder_layer_call_and_return_conditional_losses_8186346
input_57,
conv2d_37_8186315: 
conv2d_37_8186317:	-
batch_normalization_55_8186321:	-
batch_normalization_55_8186323:	-
batch_normalization_55_8186325:	-
batch_normalization_55_8186327:	,
conv2d_38_8186330:
conv2d_38_8186332:,
batch_normalization_56_8186336:,
batch_normalization_56_8186338:,
batch_normalization_56_8186340:,
batch_normalization_56_8186342:
identityЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ!conv2d_37/StatefulPartitionedCallЂ!conv2d_38/StatefulPartitionedCall
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinput_57conv2d_37_8186315conv2d_37_8186317*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100љ
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_55_8186321batch_normalization_55_8186323batch_normalization_55_8186325batch_normalization_55_8186327*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185964Г
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0conv2d_38_8186330conv2d_38_8186332*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127ј
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_56_8186336batch_normalization_56_8186338batch_normalization_56_8186340batch_normalization_56_8186342*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186040
IdentityIdentity7batch_normalization_56/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ№
NoOpNoOp/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_57
Р
b
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186682

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               u
PadPadinputsPad/paddings:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџn
IdentityIdentityPad:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187887
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

i
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8188317

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_8187867
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
г
8__inference_batch_normalization_58_layer_call_fn_8188742

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186626
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188453

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186040

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
Ў
)__inference_Decoder_layer_call_fn_8186749
input_58"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinput_58unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186722
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_58
Ф
b
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188664

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               v
PadPadinputsPad/paddings:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџo
IdentityIdentityPad:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
г
8__inference_batch_normalization_56_layer_call_fn_8188422

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186040
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
+
м
D__inference_Decoder_layer_call_and_return_conditional_losses_8186850

inputs6
conv2d_transpose_27_8186819:*
conv2d_transpose_27_8186821:	-
batch_normalization_57_8186824:	-
batch_normalization_57_8186826:	-
batch_normalization_57_8186828:	-
batch_normalization_57_8186830:	6
conv2d_transpose_28_8186835:)
conv2d_transpose_28_8186837:,
batch_normalization_58_8186840:,
batch_normalization_58_8186842:,
batch_normalization_58_8186844:,
batch_normalization_58_8186846:
identityЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallц
 up_sampling2d_27/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396ћ
lambda_27/PartitionedCallPartitionedCall)up_sampling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186780й
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"lambda_27/PartitionedCall:output:0conv2d_transpose_27_8186819conv2d_transpose_27_8186821*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453Й
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_57_8186824batch_normalization_57_8186826batch_normalization_57_8186828batch_normalization_57_8186830*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186513
 up_sampling2d_28/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540ќ
lambda_28/PartitionedCallPartitionedCall)up_sampling2d_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186763и
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall"lambda_28/PartitionedCall:output:0conv2d_transpose_28_8186835conv2d_transpose_28_8186837*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597И
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_58_8186840batch_normalization_58_8186842batch_normalization_58_8186844batch_normalization_58_8186846*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186657 
IdentityIdentity7batch_normalization_58/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187518

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:%

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187036
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
N
2__inference_max_pooling2d_34_layer_call_fn_8188404

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ђ
+__inference_conv2d_37_layer_call_fn_8188296

inputs"
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
Ќ
)__inference_Decoder_layer_call_fn_8188119

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186850
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_8187862
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
­
L
$__inference__update_step_xla_8187902
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
г
8__inference_batch_normalization_58_layer_call_fn_8188755

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186657
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Пј
Ё$
"__inference__wrapped_model_8185930
input_59g
Lfully_convolutional_ae_stft_encoder_conv2d_37_conv2d_readvariableop_resource:\
Mfully_convolutional_ae_stft_encoder_conv2d_37_biasadd_readvariableop_resource:	a
Rfully_convolutional_ae_stft_encoder_batch_normalization_55_readvariableop_resource:	c
Tfully_convolutional_ae_stft_encoder_batch_normalization_55_readvariableop_1_resource:	r
cfully_convolutional_ae_stft_encoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:	t
efully_convolutional_ae_stft_encoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:	g
Lfully_convolutional_ae_stft_encoder_conv2d_38_conv2d_readvariableop_resource:[
Mfully_convolutional_ae_stft_encoder_conv2d_38_biasadd_readvariableop_resource:`
Rfully_convolutional_ae_stft_encoder_batch_normalization_56_readvariableop_resource:b
Tfully_convolutional_ae_stft_encoder_batch_normalization_56_readvariableop_1_resource:q
cfully_convolutional_ae_stft_encoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource:s
efully_convolutional_ae_stft_encoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:{
`fully_convolutional_ae_stft_decoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource:f
Wfully_convolutional_ae_stft_decoder_conv2d_transpose_27_biasadd_readvariableop_resource:	a
Rfully_convolutional_ae_stft_decoder_batch_normalization_57_readvariableop_resource:	c
Tfully_convolutional_ae_stft_decoder_batch_normalization_57_readvariableop_1_resource:	r
cfully_convolutional_ae_stft_decoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource:	t
efully_convolutional_ae_stft_decoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:	{
`fully_convolutional_ae_stft_decoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource:e
Wfully_convolutional_ae_stft_decoder_conv2d_transpose_28_biasadd_readvariableop_resource:`
Rfully_convolutional_ae_stft_decoder_batch_normalization_58_readvariableop_resource:b
Tfully_convolutional_ae_stft_decoder_batch_normalization_58_readvariableop_1_resource:q
cfully_convolutional_ae_stft_decoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource:s
efully_convolutional_ae_stft_decoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource:
identityЂZFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ЂIFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOpЂKFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_1ЂZFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ЂIFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOpЂKFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_1ЂNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpЂWFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpЂWFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂZFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ЂIFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOpЂKFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_1ЂZFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ЂIFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOpЂKFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_1ЂDFully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOpЂCFully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOpЂDFully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOpЂCFully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOpй
CFully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOpReadVariableOpLfully_convolutional_ae_stft_encoder_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0љ
4Fully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2DConv2Dinput_59KFully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides
Я
DFully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOpReadVariableOpMfully_convolutional_ae_stft_encoder_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
5Fully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAddBiasAdd=Fully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D:output:0LFully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?Г
1Fully_Convolutional_AE_STFT/Encoder/conv2d_37/EluElu>Fully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?і
<Fully_Convolutional_AE_STFT/Encoder/max_pooling2d_33/MaxPoolMaxPool?Fully_Convolutional_AE_STFT/Encoder/conv2d_37/Elu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
й
IFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOpReadVariableOpRfully_convolutional_ae_stft_encoder_batch_normalization_55_readvariableop_resource*
_output_shapes	
:*
dtype0н
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_1ReadVariableOpTfully_convolutional_ae_stft_encoder_batch_normalization_55_readvariableop_1_resource*
_output_shapes	
:*
dtype0ћ
ZFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpcfully_convolutional_ae_stft_encoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0џ
\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefully_convolutional_ae_stft_encoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ё
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3EFully_Convolutional_AE_STFT/Encoder/max_pooling2d_33/MaxPool:output:0QFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp:value:0SFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_1:value:0bFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0dFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( й
CFully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOpReadVariableOpLfully_convolutional_ae_stft_encoder_conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0П
4Fully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2DConv2DOFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3:y:0KFully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ю
DFully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOpReadVariableOpMfully_convolutional_ae_stft_encoder_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
5Fully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAddBiasAdd=Fully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D:output:0LFully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџВ
1Fully_Convolutional_AE_STFT/Encoder/conv2d_38/EluElu>Fully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџѕ
<Fully_Convolutional_AE_STFT/Encoder/max_pooling2d_34/MaxPoolMaxPool?Fully_Convolutional_AE_STFT/Encoder/conv2d_38/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
и
IFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOpReadVariableOpRfully_convolutional_ae_stft_encoder_batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype0м
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_1ReadVariableOpTfully_convolutional_ae_stft_encoder_batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype0њ
ZFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOpcfully_convolutional_ae_stft_encoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ў
\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefully_convolutional_ae_stft_encoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3FusedBatchNormV3EFully_Convolutional_AE_STFT/Encoder/max_pooling2d_34/MaxPool:output:0QFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp:value:0SFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_1:value:0bFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0dFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
:Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
<Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      №
8Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/mulMulCFully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/Const:output:0EFully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/Const_1:output:0*
T0*
_output_shapes
:Э
QFully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/resize/ResizeNearestNeighborResizeNearestNeighborOFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3:y:0<Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(Ћ
:Fully_Convolutional_AE_STFT/Decoder/lambda_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               
1Fully_Convolutional_AE_STFT/Decoder/lambda_27/PadPadbFully_Convolutional_AE_STFT/Decoder/up_sampling2d_27/resize/ResizeNearestNeighbor:resized_images:0CFully_Convolutional_AE_STFT/Decoder/lambda_27/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџЇ
=Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/ShapeShape:Fully_Convolutional_AE_STFT/Decoder/lambda_27/Pad:output:0*
T0*
_output_shapes
:
KFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
EFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_sliceStridedSliceFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/Shape:output:0TFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stack:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stack_1:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Ё
=Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stackPackNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/1:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/2:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
OFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
OFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
GFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1StridedSliceFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stack:output:0XFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stack_1:output:0XFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
WFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp`fully_convolutional_ae_stft_decoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0Ј
HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transposeConv2DBackpropInputFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/stack:output:0_Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0:Fully_Convolutional_AE_STFT/Decoder/lambda_27/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
у
NFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOpWfully_convolutional_ae_stft_decoder_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAddBiasAddQFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
;Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/EluEluHFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџй
IFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOpReadVariableOpRfully_convolutional_ae_stft_decoder_batch_normalization_57_readvariableop_resource*
_output_shapes	
:*
dtype0н
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_1ReadVariableOpTfully_convolutional_ae_stft_decoder_batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:*
dtype0ћ
ZFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpcfully_convolutional_ae_stft_decoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0џ
\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefully_convolutional_ae_stft_decoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ѕ
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3FusedBatchNormV3IFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/Elu:activations:0QFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp:value:0SFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_1:value:0bFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0dFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
:Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
<Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      №
8Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/mulMulCFully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/Const:output:0EFully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/Const_1:output:0*
T0*
_output_shapes
:Ю
QFully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/resize/ResizeNearestNeighborResizeNearestNeighborOFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3:y:0<Fully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ>*
half_pixel_centers(Ћ
:Fully_Convolutional_AE_STFT/Decoder/lambda_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               
1Fully_Convolutional_AE_STFT/Decoder/lambda_28/PadPadbFully_Convolutional_AE_STFT/Decoder/up_sampling2d_28/resize/ResizeNearestNeighbor:resized_images:0CFully_Convolutional_AE_STFT/Decoder/lambda_28/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?Ї
=Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/ShapeShape:Fully_Convolutional_AE_STFT/Decoder/lambda_28/Pad:output:0*
T0*
_output_shapes
:
KFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
EFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_sliceStridedSliceFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/Shape:output:0TFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stack:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stack_1:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value
B :
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ё
=Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stackPackNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/1:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/2:output:0HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:
MFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
OFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
OFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
GFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1StridedSliceFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stack:output:0XFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stack_1:output:0XFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
WFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp`fully_convolutional_ae_stft_decoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0Ј
HFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transposeConv2DBackpropInputFFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/stack:output:0_Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0:Fully_Convolutional_AE_STFT/Decoder/lambda_28/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
т
NFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOpWfully_convolutional_ae_stft_decoder_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
?Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAddBiasAddQFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose:output:0VFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@Ч
;Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/EluEluHFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@и
IFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOpReadVariableOpRfully_convolutional_ae_stft_decoder_batch_normalization_58_readvariableop_resource*
_output_shapes
:*
dtype0м
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_1ReadVariableOpTfully_convolutional_ae_stft_decoder_batch_normalization_58_readvariableop_1_resource*
_output_shapes
:*
dtype0њ
ZFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOpcfully_convolutional_ae_stft_decoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ў
\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefully_convolutional_ae_stft_decoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ё
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3FusedBatchNormV3IFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/Elu:activations:0QFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp:value:0SFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_1:value:0bFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0dFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ@:::::*
epsilon%o:*
is_training( Ї
IdentityIdentityOFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp[^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp]^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1J^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOpL^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_1[^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp]^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1J^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOpL^Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_1O^Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpX^Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpO^Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpX^Fully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp[^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp]^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1J^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOpL^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_1[^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp]^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1J^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOpL^Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_1E^Fully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOpD^Fully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOpE^Fully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOpD^Fully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2И
ZFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpZFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp2М
\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12
IFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOpIFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp2
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_1KFully_Convolutional_AE_STFT/Decoder/batch_normalization_57/ReadVariableOp_12И
ZFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpZFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp2М
\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1\Fully_Convolutional_AE_STFT/Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12
IFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOpIFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp2
KFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_1KFully_Convolutional_AE_STFT/Decoder/batch_normalization_58/ReadVariableOp_12 
NFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2В
WFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpWFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2 
NFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpNFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2В
WFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpWFully_Convolutional_AE_STFT/Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2И
ZFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpZFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2М
\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12
IFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOpIFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp2
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_1KFully_Convolutional_AE_STFT/Encoder/batch_normalization_55/ReadVariableOp_12И
ZFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpZFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp2М
\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1\Fully_Convolutional_AE_STFT/Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12
IFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOpIFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp2
KFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_1KFully_Convolutional_AE_STFT/Encoder/batch_normalization_56/ReadVariableOp_12
DFully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOpDFully_Convolutional_AE_STFT/Encoder/conv2d_37/BiasAdd/ReadVariableOp2
CFully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOpCFully_Convolutional_AE_STFT/Encoder/conv2d_37/Conv2D/ReadVariableOp2
DFully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOpDFully_Convolutional_AE_STFT/Encoder/conv2d_38/BiasAdd/ReadVariableOp2
CFully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOpCFully_Convolutional_AE_STFT/Encoder/conv2d_38/Conv2D/ReadVariableOp:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59
Ф
b
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188670

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               v
PadPadinputsPad/paddings:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџo
IdentityIdentityPad:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188613

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_lambda_28_layer_call_fn_8188653

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186705{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_lambda_27_layer_call_fn_8188493

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186682z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186626

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П
N
2__inference_up_sampling2d_27_layer_call_fn_8188476

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127

inputs9
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџV
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџh
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_lambda_28_layer_call_fn_8188658

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186763{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ќ
)__inference_Encoder_layer_call_fn_8187965

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186256w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Р
b
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186780

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               u
PadPadinputsPad/paddings:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџn
IdentityIdentityPad:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
г
8__inference_batch_normalization_56_layer_call_fn_8188435

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186071
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185964

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


%__inference_signature_wrapper_8187465
input_59"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:%

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_59unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_8185930x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59
в
Ќ
5__inference_conv2d_transpose_27_layer_call_fn_8188519

inputs"
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Є+
о
D__inference_Decoder_layer_call_and_return_conditional_losses_8186978
input_586
conv2d_transpose_27_8186947:*
conv2d_transpose_27_8186949:	-
batch_normalization_57_8186952:	-
batch_normalization_57_8186954:	-
batch_normalization_57_8186956:	-
batch_normalization_57_8186958:	6
conv2d_transpose_28_8186963:)
conv2d_transpose_28_8186965:,
batch_normalization_58_8186968:,
batch_normalization_58_8186970:,
batch_normalization_58_8186972:,
batch_normalization_58_8186974:
identityЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallш
 up_sampling2d_27/PartitionedCallPartitionedCallinput_58*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396ћ
lambda_27/PartitionedCallPartitionedCall)up_sampling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186780й
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"lambda_27/PartitionedCall:output:0conv2d_transpose_27_8186947conv2d_transpose_27_8186949*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453Й
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_57_8186952batch_normalization_57_8186954batch_normalization_57_8186956batch_normalization_57_8186958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186513
 up_sampling2d_28/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540ќ
lambda_28/PartitionedCallPartitionedCall)up_sampling2d_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186763и
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall"lambda_28/PartitionedCall:output:0conv2d_transpose_28_8186963conv2d_transpose_28_8186965*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597И
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_58_8186968batch_normalization_58_8186970batch_normalization_58_8186972batch_normalization_58_8186974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186657 
IdentityIdentity7batch_normalization_58/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_58
§*

P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453

inputsC
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :U
subSubstrided_slice_1:output:0sub/y:output:0*
T0*
_output_shapes
: G
mul/yConst*
_output_shapes
: *
dtype0*
value	B :D
mulMulsub:z:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : H
sub_1Subadd:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	sub_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_2AddV2	mul_1:z:0add_2/y:output:0*
T0*
_output_shapes
: I
sub_3/yConst*
_output_shapes
: *
dtype0*
value	B : J
sub_3Sub	add_2:z:0sub_3/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_3AddV2	sub_3:z:0add_3/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :{
stackPackstrided_slice:output:0	add_1:z:0	add_3:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0о
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б
Ћ
5__inference_conv2d_transpose_28_layer_call_fn_8188679

inputs"
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ+
м
D__inference_Decoder_layer_call_and_return_conditional_losses_8186722

inputs6
conv2d_transpose_27_8186684:*
conv2d_transpose_27_8186686:	-
batch_normalization_57_8186689:	-
batch_normalization_57_8186691:	-
batch_normalization_57_8186693:	-
batch_normalization_57_8186695:	6
conv2d_transpose_28_8186707:)
conv2d_transpose_28_8186709:,
batch_normalization_58_8186712:,
batch_normalization_58_8186714:,
batch_normalization_58_8186716:,
batch_normalization_58_8186718:
identityЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallц
 up_sampling2d_27/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396ћ
lambda_27/PartitionedCallPartitionedCall)up_sampling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186682й
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"lambda_27/PartitionedCall:output:0conv2d_transpose_27_8186684conv2d_transpose_27_8186686*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453Л
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_57_8186689batch_normalization_57_8186691batch_normalization_57_8186693batch_normalization_57_8186695*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186482
 up_sampling2d_28/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540ќ
lambda_28/PartitionedCallPartitionedCall)up_sampling2d_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186705и
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall"lambda_28/PartitionedCall:output:0conv2d_transpose_28_8186707conv2d_transpose_28_8186709*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597К
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_58_8186712batch_normalization_58_8186714batch_normalization_58_8186716batch_normalization_58_8186718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186626 
IdentityIdentity7batch_normalization_58/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
с
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187036

inputs*
encoder_8186985:
encoder_8186987:	
encoder_8186989:	
encoder_8186991:	
encoder_8186993:	
encoder_8186995:	*
encoder_8186997:
encoder_8186999:
encoder_8187001:
encoder_8187003:
encoder_8187005:
encoder_8187007:*
decoder_8187010:
decoder_8187012:	
decoder_8187014:	
decoder_8187016:	
decoder_8187018:	
decoder_8187020:	*
decoder_8187022:
decoder_8187024:
decoder_8187026:
decoder_8187028:
decoder_8187030:
decoder_8187032:
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallИ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_8186985encoder_8186987encoder_8186989encoder_8186991encoder_8186993encoder_8186995encoder_8186997encoder_8186999encoder_8187001encoder_8187003encoder_8187005encoder_8187007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186144ь
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_8187010decoder_8187012decoder_8187014decoder_8187016decoder_8187018decoder_8187020decoder_8187022decoder_8187024decoder_8187026decoder_8187028decoder_8187030decoder_8187032*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186722
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_8187857
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
о
Ђ
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186482

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186071

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј+
о
D__inference_Decoder_layer_call_and_return_conditional_losses_8186942
input_586
conv2d_transpose_27_8186911:*
conv2d_transpose_27_8186913:	-
batch_normalization_57_8186916:	-
batch_normalization_57_8186918:	-
batch_normalization_57_8186920:	-
batch_normalization_57_8186922:	6
conv2d_transpose_28_8186927:)
conv2d_transpose_28_8186929:,
batch_normalization_58_8186932:,
batch_normalization_58_8186934:,
batch_normalization_58_8186936:,
batch_normalization_58_8186938:
identityЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallш
 up_sampling2d_27/PartitionedCallPartitionedCallinput_58*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8186396ћ
lambda_27/PartitionedCallPartitionedCall)up_sampling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186682й
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"lambda_27/PartitionedCall:output:0conv2d_transpose_27_8186911conv2d_transpose_27_8186913*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8186453Л
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_57_8186916batch_normalization_57_8186918batch_normalization_57_8186920batch_normalization_57_8186922*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186482
 up_sampling2d_28/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540ќ
lambda_28/PartitionedCallPartitionedCall)up_sampling2d_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186705и
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall"lambda_28/PartitionedCall:output:0conv2d_transpose_28_8186927conv2d_transpose_28_8186929*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597К
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_58_8186932batch_normalization_58_8186934batch_normalization_58_8186936batch_normalization_58_8186938*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186626 
IdentityIdentity7batch_normalization_58/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_58
П
N
2__inference_up_sampling2d_28_layer_call_fn_8188636

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187842
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ъj
Ў
D__inference_Decoder_layer_call_and_return_conditional_losses_8188203

inputsW
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource:B
3conv2d_transpose_27_biasadd_readvariableop_resource:	=
.batch_normalization_57_readvariableop_resource:	?
0batch_normalization_57_readvariableop_1_resource:	N
?batch_normalization_57_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_28_biasadd_readvariableop_resource:<
.batch_normalization_58_readvariableop_resource:>
0batch_normalization_58_readvariableop_1_resource:M
?batch_normalization_58_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource:
identityЂ6batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_57/ReadVariableOpЂ'batch_normalization_57/ReadVariableOp_1Ђ6batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_58/ReadVariableOpЂ'batch_normalization_58/ReadVariableOp_1Ђ*conv2d_transpose_27/BiasAdd/ReadVariableOpЂ3conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_28/BiasAdd/ReadVariableOpЂ3conv2d_transpose_28/conv2d_transpose/ReadVariableOpg
up_sampling2d_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_27/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_27/mulMulup_sampling2d_27/Const:output:0!up_sampling2d_27/Const_1:output:0*
T0*
_output_shapes
:М
-up_sampling2d_27/resize/ResizeNearestNeighborResizeNearestNeighborinputsup_sampling2d_27/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(
lambda_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Џ
lambda_27/PadPad>up_sampling2d_27/resize/ResizeNearestNeighbor:resized_images:0lambda_27/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ_
conv2d_transpose_27/ShapeShapelambda_27/Pad:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0lambda_27/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_transpose_27/EluElu$conv2d_transpose_27/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_27/Elu:activations:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( g
up_sampling2d_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_28/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_28/mulMulup_sampling2d_28/Const:output:0!up_sampling2d_28/Const_1:output:0*
T0*
_output_shapes
:т
-up_sampling2d_28/resize/ResizeNearestNeighborResizeNearestNeighbor+batch_normalization_57/FusedBatchNormV3:y:0up_sampling2d_28/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ>*
half_pixel_centers(
lambda_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               А
lambda_28/PadPad>up_sampling2d_28/resize/ResizeNearestNeighbor:resized_images:0lambda_28/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?_
conv2d_transpose_28/ShapeShapelambda_28/Pad:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@^
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0lambda_28/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides

*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@
conv2d_transpose_28/EluElu$conv2d_transpose_28/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Щ
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_28/Elu:activations:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ@:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_58/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp7^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_17^batch_normalization_58/FusedBatchNormV3/ReadVariableOp9^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_58/ReadVariableOp(^batch_normalization_58/ReadVariableOp_1+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12p
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp6batch_normalization_58/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_18batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_58/ReadVariableOp%batch_normalization_58/ReadVariableOp2R
'batch_normalization_58/ReadVariableOp_1'batch_normalization_58/ReadVariableOp_12X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8188409

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8186540

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187882
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
y
в
D__inference_Decoder_layer_call_and_return_conditional_losses_8188287

inputsW
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource:B
3conv2d_transpose_27_biasadd_readvariableop_resource:	=
.batch_normalization_57_readvariableop_resource:	?
0batch_normalization_57_readvariableop_1_resource:	N
?batch_normalization_57_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_28_biasadd_readvariableop_resource:<
.batch_normalization_58_readvariableop_resource:>
0batch_normalization_58_readvariableop_1_resource:M
?batch_normalization_58_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource:
identityЂ%batch_normalization_57/AssignNewValueЂ'batch_normalization_57/AssignNewValue_1Ђ6batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_57/ReadVariableOpЂ'batch_normalization_57/ReadVariableOp_1Ђ%batch_normalization_58/AssignNewValueЂ'batch_normalization_58/AssignNewValue_1Ђ6batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_58/ReadVariableOpЂ'batch_normalization_58/ReadVariableOp_1Ђ*conv2d_transpose_27/BiasAdd/ReadVariableOpЂ3conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_28/BiasAdd/ReadVariableOpЂ3conv2d_transpose_28/conv2d_transpose/ReadVariableOpg
up_sampling2d_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_27/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_27/mulMulup_sampling2d_27/Const:output:0!up_sampling2d_27/Const_1:output:0*
T0*
_output_shapes
:М
-up_sampling2d_27/resize/ResizeNearestNeighborResizeNearestNeighborinputsup_sampling2d_27/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(
lambda_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Џ
lambda_27/PadPad>up_sampling2d_27/resize/ResizeNearestNeighbor:resized_images:0lambda_27/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ_
conv2d_transpose_27/ShapeShapelambda_27/Pad:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0lambda_27/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_transpose_27/EluElu$conv2d_transpose_27/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_27/Elu:activations:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_57/AssignNewValueAssignVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource4batch_normalization_57/FusedBatchNormV3:batch_mean:07^batch_normalization_57/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_57/AssignNewValue_1AssignVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_57/FusedBatchNormV3:batch_variance:09^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(g
up_sampling2d_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_28/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_28/mulMulup_sampling2d_28/Const:output:0!up_sampling2d_28/Const_1:output:0*
T0*
_output_shapes
:т
-up_sampling2d_28/resize/ResizeNearestNeighborResizeNearestNeighbor+batch_normalization_57/FusedBatchNormV3:y:0up_sampling2d_28/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ>*
half_pixel_centers(
lambda_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               А
lambda_28/PadPad>up_sampling2d_28/resize/ResizeNearestNeighbor:resized_images:0lambda_28/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?_
conv2d_transpose_28/ShapeShapelambda_28/Pad:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@^
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0lambda_28/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides

*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@
conv2d_transpose_28/EluElu$conv2d_transpose_28/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0з
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_28/Elu:activations:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ@:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_58/AssignNewValueAssignVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource4batch_normalization_58/FusedBatchNormV3:batch_mean:07^batch_normalization_58/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_58/AssignNewValue_1AssignVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_58/FusedBatchNormV3:batch_variance:09^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_58/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@М
NoOpNoOp&^batch_normalization_57/AssignNewValue(^batch_normalization_57/AssignNewValue_17^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_1&^batch_normalization_58/AssignNewValue(^batch_normalization_58/AssignNewValue_17^batch_normalization_58/FusedBatchNormV3/ReadVariableOp9^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_58/ReadVariableOp(^batch_normalization_58/ReadVariableOp_1+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 2N
%batch_normalization_57/AssignNewValue%batch_normalization_57/AssignNewValue2R
'batch_normalization_57/AssignNewValue_1'batch_normalization_57/AssignNewValue_12p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12N
%batch_normalization_58/AssignNewValue%batch_normalization_58/AssignNewValue2R
'batch_normalization_58/AssignNewValue_1'batch_normalization_58/AssignNewValue_12p
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp6batch_normalization_58/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_18batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_58/ReadVariableOp%batch_normalization_58/ReadVariableOp2R
'batch_normalization_58/ReadVariableOp_1'batch_normalization_58/ReadVariableOp_12X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186513

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Й
Ў
)__inference_Decoder_layer_call_fn_8186906
input_58"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_58unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186850
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_58
ј*

P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8186597

inputsC
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :U
subSubstrided_slice_1:output:0sub/y:output:0*
T0*
_output_shapes
: G
mul/yConst*
_output_shapes
: *
dtype0*
value	B :D
mulMulsub:z:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : H
sub_1Subadd:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	sub_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_2AddV2	mul_1:z:0add_2/y:output:0*
T0*
_output_shapes
: I
sub_3/yConst*
_output_shapes
: *
dtype0*
value	B : J
sub_3Sub	add_2:z:0sub_3/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_3AddV2	sub_3:z:0add_3/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :{
stackPackstrided_slice:output:0	add_1:z:0	add_3:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188471

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8186657

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж;
Ж
D__inference_Encoder_layer_call_and_return_conditional_losses_8188013

inputsC
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	=
.batch_normalization_55_readvariableop_resource:	?
0batch_normalization_55_readvariableop_1_resource:	N
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_38_conv2d_readvariableop_resource:7
)conv2d_38_biasadd_readvariableop_resource:<
.batch_normalization_56_readvariableop_resource:>
0batch_normalization_56_readvariableop_1_resource:M
?batch_normalization_56_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:
identityЂ6batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_55/ReadVariableOpЂ'batch_normalization_55/ReadVariableOp_1Ђ6batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_56/ReadVariableOpЂ'batch_normalization_56/ReadVariableOp_1Ђ conv2d_37/BiasAdd/ReadVariableOpЂconv2d_37/Conv2D/ReadVariableOpЂ conv2d_38/BiasAdd/ReadVariableOpЂconv2d_38/Conv2D/ReadVariableOp
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Џ
conv2d_37/Conv2DConv2Dinputs'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides

 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?k
conv2d_37/EluEluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?Ў
max_pooling2d_33/MaxPoolMaxPoolconv2d_37/Elu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0г
conv2d_38/Conv2DConv2D+batch_normalization_55/FusedBatchNormV3:y:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_38/EluEluconv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ­
max_pooling2d_34/MaxPoolMaxPoolconv2d_38/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ф
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_56/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџм
NoOpNoOp7^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_1!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188773

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ў
)__inference_Encoder_layer_call_fn_8186312
input_57"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinput_57unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186256w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_57
Й
у
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187408
input_59*
encoder_8187357:
encoder_8187359:	
encoder_8187361:	
encoder_8187363:	
encoder_8187365:	
encoder_8187367:	*
encoder_8187369:
encoder_8187371:
encoder_8187373:
encoder_8187375:
encoder_8187377:
encoder_8187379:*
decoder_8187382:
decoder_8187384:	
decoder_8187386:	
decoder_8187388:	
decoder_8187390:	
decoder_8187392:	*
decoder_8187394:
decoder_8187396:
decoder_8187398:
decoder_8187400:
decoder_8187402:
decoder_8187404:
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallЖ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_59encoder_8187357encoder_8187359encoder_8187361encoder_8187363encoder_8187365encoder_8187367encoder_8187369encoder_8187371encoder_8187373encoder_8187375encoder_8187377encoder_8187379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186256ш
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_8187382decoder_8187384decoder_8187386decoder_8187388decoder_8187390decoder_8187392decoder_8187394decoder_8187396decoder_8187398decoder_8187400decoder_8187402decoder_8187404*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186850
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59
є
Ё
+__inference_conv2d_38_layer_call_fn_8188388

inputs"
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


F__inference_conv2d_38_layer_call_and_return_conditional_losses_8188399

inputs9
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџV
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџh
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8188488

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕж

X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187827

inputsK
0encoder_conv2d_37_conv2d_readvariableop_resource:@
1encoder_conv2d_37_biasadd_readvariableop_resource:	E
6encoder_batch_normalization_55_readvariableop_resource:	G
8encoder_batch_normalization_55_readvariableop_1_resource:	V
Gencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:	X
Iencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:	K
0encoder_conv2d_38_conv2d_readvariableop_resource:?
1encoder_conv2d_38_biasadd_readvariableop_resource:D
6encoder_batch_normalization_56_readvariableop_resource:F
8encoder_batch_normalization_56_readvariableop_1_resource:U
Gencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:_
Ddecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource:J
;decoder_conv2d_transpose_27_biasadd_readvariableop_resource:	E
6decoder_batch_normalization_57_readvariableop_resource:	G
8decoder_batch_normalization_57_readvariableop_1_resource:	V
Gdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource:	X
Idecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:	_
Ddecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_28_biasadd_readvariableop_resource:D
6decoder_batch_normalization_58_readvariableop_resource:F
8decoder_batch_normalization_58_readvariableop_1_resource:U
Gdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource:W
Idecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource:
identityЂ-Decoder/batch_normalization_57/AssignNewValueЂ/Decoder/batch_normalization_57/AssignNewValue_1Ђ>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ-Decoder/batch_normalization_57/ReadVariableOpЂ/Decoder/batch_normalization_57/ReadVariableOp_1Ђ-Decoder/batch_normalization_58/AssignNewValueЂ/Decoder/batch_normalization_58/AssignNewValue_1Ђ>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ-Decoder/batch_normalization_58/ReadVariableOpЂ/Decoder/batch_normalization_58/ReadVariableOp_1Ђ2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpЂ;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpЂ;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂ-Encoder/batch_normalization_55/AssignNewValueЂ/Encoder/batch_normalization_55/AssignNewValue_1Ђ>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ-Encoder/batch_normalization_55/ReadVariableOpЂ/Encoder/batch_normalization_55/ReadVariableOp_1Ђ-Encoder/batch_normalization_56/AssignNewValueЂ/Encoder/batch_normalization_56/AssignNewValue_1Ђ>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ-Encoder/batch_normalization_56/ReadVariableOpЂ/Encoder/batch_normalization_56/ReadVariableOp_1Ђ(Encoder/conv2d_37/BiasAdd/ReadVariableOpЂ'Encoder/conv2d_37/Conv2D/ReadVariableOpЂ(Encoder/conv2d_38/BiasAdd/ReadVariableOpЂ'Encoder/conv2d_38/Conv2D/ReadVariableOpЁ
'Encoder/conv2d_37/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0П
Encoder/conv2d_37/Conv2DConv2Dinputs/Encoder/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides

(Encoder/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
Encoder/conv2d_37/BiasAddBiasAdd!Encoder/conv2d_37/Conv2D:output:00Encoder/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?{
Encoder/conv2d_37/EluElu"Encoder/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?О
 Encoder/max_pooling2d_33/MaxPoolMaxPool#Encoder/conv2d_37/Elu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Ё
-Encoder/batch_normalization_55/ReadVariableOpReadVariableOp6encoder_batch_normalization_55_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/Encoder/batch_normalization_55/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_55_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
/Encoder/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3)Encoder/max_pooling2d_33/MaxPool:output:05Encoder/batch_normalization_55/ReadVariableOp:value:07Encoder/batch_normalization_55/ReadVariableOp_1:value:0FEncoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0HEncoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Т
-Encoder/batch_normalization_55/AssignNewValueAssignVariableOpGencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource<Encoder/batch_normalization_55/FusedBatchNormV3:batch_mean:0?^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ь
/Encoder/batch_normalization_55/AssignNewValue_1AssignVariableOpIencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource@Encoder/batch_normalization_55/FusedBatchNormV3:batch_variance:0A^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ё
'Encoder/conv2d_38/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ы
Encoder/conv2d_38/Conv2DConv2D3Encoder/batch_normalization_55/FusedBatchNormV3:y:0/Encoder/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

(Encoder/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
Encoder/conv2d_38/BiasAddBiasAdd!Encoder/conv2d_38/Conv2D:output:00Encoder/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџz
Encoder/conv2d_38/EluElu"Encoder/conv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџН
 Encoder/max_pooling2d_34/MaxPoolMaxPool#Encoder/conv2d_38/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
 
-Encoder/batch_normalization_56/ReadVariableOpReadVariableOp6encoder_batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype0Є
/Encoder/batch_normalization_56/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
/Encoder/batch_normalization_56/FusedBatchNormV3FusedBatchNormV3)Encoder/max_pooling2d_34/MaxPool:output:05Encoder/batch_normalization_56/ReadVariableOp:value:07Encoder/batch_normalization_56/ReadVariableOp_1:value:0FEncoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0HEncoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Т
-Encoder/batch_normalization_56/AssignNewValueAssignVariableOpGencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource<Encoder/batch_normalization_56/FusedBatchNormV3:batch_mean:0?^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ь
/Encoder/batch_normalization_56/AssignNewValue_1AssignVariableOpIencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource@Encoder/batch_normalization_56/FusedBatchNormV3:batch_variance:0A^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(o
Decoder/up_sampling2d_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 Decoder/up_sampling2d_27/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_27/mulMul'Decoder/up_sampling2d_27/Const:output:0)Decoder/up_sampling2d_27/Const_1:output:0*
T0*
_output_shapes
:љ
5Decoder/up_sampling2d_27/resize/ResizeNearestNeighborResizeNearestNeighbor3Encoder/batch_normalization_56/FusedBatchNormV3:y:0 Decoder/up_sampling2d_27/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(
Decoder/lambda_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Ч
Decoder/lambda_27/PadPadFDecoder/up_sampling2d_27/resize/ResizeNearestNeighbor:resized_images:0'Decoder/lambda_27/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџo
!Decoder/conv2d_transpose_27/ShapeShapeDecoder/lambda_27/Pad:output:0*
T0*
_output_shapes
:y
/Decoder/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1Decoder/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1Decoder/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)Decoder/conv2d_transpose_27/strided_sliceStridedSlice*Decoder/conv2d_transpose_27/Shape:output:08Decoder/conv2d_transpose_27/strided_slice/stack:output:0:Decoder/conv2d_transpose_27/strided_slice/stack_1:output:0:Decoder/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#Decoder/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#Decoder/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#Decoder/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!Decoder/conv2d_transpose_27/stackPack2Decoder/conv2d_transpose_27/strided_slice:output:0,Decoder/conv2d_transpose_27/stack/1:output:0,Decoder/conv2d_transpose_27/stack/2:output:0,Decoder/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:{
1Decoder/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Decoder/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Decoder/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+Decoder/conv2d_transpose_27/strided_slice_1StridedSlice*Decoder/conv2d_transpose_27/stack:output:0:Decoder/conv2d_transpose_27/strided_slice_1/stack:output:0<Decoder/conv2d_transpose_27/strided_slice_1/stack_1:output:0<Decoder/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0И
,Decoder/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput*Decoder/conv2d_transpose_27/stack:output:0CDecoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0Decoder/lambda_27/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ћ
2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#Decoder/conv2d_transpose_27/BiasAddBiasAdd5Decoder/conv2d_transpose_27/conv2d_transpose:output:0:Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
Decoder/conv2d_transpose_27/EluElu,Decoder/conv2d_transpose_27/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЁ
-Decoder/batch_normalization_57/ReadVariableOpReadVariableOp6decoder_batch_normalization_57_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/Decoder/batch_normalization_57/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
/Decoder/batch_normalization_57/FusedBatchNormV3FusedBatchNormV3-Decoder/conv2d_transpose_27/Elu:activations:05Decoder/batch_normalization_57/ReadVariableOp:value:07Decoder/batch_normalization_57/ReadVariableOp_1:value:0FDecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0HDecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Т
-Decoder/batch_normalization_57/AssignNewValueAssignVariableOpGdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource<Decoder/batch_normalization_57/FusedBatchNormV3:batch_mean:0?^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ь
/Decoder/batch_normalization_57/AssignNewValue_1AssignVariableOpIdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource@Decoder/batch_normalization_57/FusedBatchNormV3:batch_variance:0A^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(o
Decoder/up_sampling2d_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 Decoder/up_sampling2d_28/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_28/mulMul'Decoder/up_sampling2d_28/Const:output:0)Decoder/up_sampling2d_28/Const_1:output:0*
T0*
_output_shapes
:њ
5Decoder/up_sampling2d_28/resize/ResizeNearestNeighborResizeNearestNeighbor3Decoder/batch_normalization_57/FusedBatchNormV3:y:0 Decoder/up_sampling2d_28/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ>*
half_pixel_centers(
Decoder/lambda_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Ш
Decoder/lambda_28/PadPadFDecoder/up_sampling2d_28/resize/ResizeNearestNeighbor:resized_images:0'Decoder/lambda_28/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?o
!Decoder/conv2d_transpose_28/ShapeShapeDecoder/lambda_28/Pad:output:0*
T0*
_output_shapes
:y
/Decoder/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1Decoder/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1Decoder/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)Decoder/conv2d_transpose_28/strided_sliceStridedSlice*Decoder/conv2d_transpose_28/Shape:output:08Decoder/conv2d_transpose_28/strided_slice/stack:output:0:Decoder/conv2d_transpose_28/strided_slice/stack_1:output:0:Decoder/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#Decoder/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@f
#Decoder/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value
B :e
#Decoder/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
!Decoder/conv2d_transpose_28/stackPack2Decoder/conv2d_transpose_28/strided_slice:output:0,Decoder/conv2d_transpose_28/stack/1:output:0,Decoder/conv2d_transpose_28/stack/2:output:0,Decoder/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:{
1Decoder/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Decoder/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Decoder/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+Decoder/conv2d_transpose_28/strided_slice_1StridedSlice*Decoder/conv2d_transpose_28/stack:output:0:Decoder/conv2d_transpose_28/strided_slice_1/stack:output:0<Decoder/conv2d_transpose_28/strided_slice_1/stack_1:output:0<Decoder/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0И
,Decoder/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput*Decoder/conv2d_transpose_28/stack:output:0CDecoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0Decoder/lambda_28/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
Њ
2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
#Decoder/conv2d_transpose_28/BiasAddBiasAdd5Decoder/conv2d_transpose_28/conv2d_transpose:output:0:Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@
Decoder/conv2d_transpose_28/EluElu,Decoder/conv2d_transpose_28/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@ 
-Decoder/batch_normalization_58/ReadVariableOpReadVariableOp6decoder_batch_normalization_58_readvariableop_resource*
_output_shapes
:*
dtype0Є
/Decoder/batch_normalization_58/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_58_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
/Decoder/batch_normalization_58/FusedBatchNormV3FusedBatchNormV3-Decoder/conv2d_transpose_28/Elu:activations:05Decoder/batch_normalization_58/ReadVariableOp:value:07Decoder/batch_normalization_58/ReadVariableOp_1:value:0FDecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0HDecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ@:::::*
epsilon%o:*
exponential_avg_factor%
з#<Т
-Decoder/batch_normalization_58/AssignNewValueAssignVariableOpGdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource<Decoder/batch_normalization_58/FusedBatchNormV3:batch_mean:0?^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ь
/Decoder/batch_normalization_58/AssignNewValue_1AssignVariableOpIdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource@Decoder/batch_normalization_58/FusedBatchNormV3:batch_variance:0A^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity3Decoder/batch_normalization_58/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@і
NoOpNoOp.^Decoder/batch_normalization_57/AssignNewValue0^Decoder/batch_normalization_57/AssignNewValue_1?^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpA^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1.^Decoder/batch_normalization_57/ReadVariableOp0^Decoder/batch_normalization_57/ReadVariableOp_1.^Decoder/batch_normalization_58/AssignNewValue0^Decoder/batch_normalization_58/AssignNewValue_1?^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpA^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1.^Decoder/batch_normalization_58/ReadVariableOp0^Decoder/batch_normalization_58/ReadVariableOp_13^Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp<^Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp3^Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp<^Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp.^Encoder/batch_normalization_55/AssignNewValue0^Encoder/batch_normalization_55/AssignNewValue_1?^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpA^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1.^Encoder/batch_normalization_55/ReadVariableOp0^Encoder/batch_normalization_55/ReadVariableOp_1.^Encoder/batch_normalization_56/AssignNewValue0^Encoder/batch_normalization_56/AssignNewValue_1?^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpA^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1.^Encoder/batch_normalization_56/ReadVariableOp0^Encoder/batch_normalization_56/ReadVariableOp_1)^Encoder/conv2d_37/BiasAdd/ReadVariableOp(^Encoder/conv2d_37/Conv2D/ReadVariableOp)^Encoder/conv2d_38/BiasAdd/ReadVariableOp(^Encoder/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2^
-Decoder/batch_normalization_57/AssignNewValue-Decoder/batch_normalization_57/AssignNewValue2b
/Decoder/batch_normalization_57/AssignNewValue_1/Decoder/batch_normalization_57/AssignNewValue_12
>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp2
@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12^
-Decoder/batch_normalization_57/ReadVariableOp-Decoder/batch_normalization_57/ReadVariableOp2b
/Decoder/batch_normalization_57/ReadVariableOp_1/Decoder/batch_normalization_57/ReadVariableOp_12^
-Decoder/batch_normalization_58/AssignNewValue-Decoder/batch_normalization_58/AssignNewValue2b
/Decoder/batch_normalization_58/AssignNewValue_1/Decoder/batch_normalization_58/AssignNewValue_12
>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp2
@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12^
-Decoder/batch_normalization_58/ReadVariableOp-Decoder/batch_normalization_58/ReadVariableOp2b
/Decoder/batch_normalization_58/ReadVariableOp_1/Decoder/batch_normalization_58/ReadVariableOp_12h
2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2z
;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2h
2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2z
;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2^
-Encoder/batch_normalization_55/AssignNewValue-Encoder/batch_normalization_55/AssignNewValue2b
/Encoder/batch_normalization_55/AssignNewValue_1/Encoder/batch_normalization_55/AssignNewValue_12
>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2
@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12^
-Encoder/batch_normalization_55/ReadVariableOp-Encoder/batch_normalization_55/ReadVariableOp2b
/Encoder/batch_normalization_55/ReadVariableOp_1/Encoder/batch_normalization_55/ReadVariableOp_12^
-Encoder/batch_normalization_56/AssignNewValue-Encoder/batch_normalization_56/AssignNewValue2b
/Encoder/batch_normalization_56/AssignNewValue_1/Encoder/batch_normalization_56/AssignNewValue_12
>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp2
@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12^
-Encoder/batch_normalization_56/ReadVariableOp-Encoder/batch_normalization_56/ReadVariableOp2b
/Encoder/batch_normalization_56/ReadVariableOp_1/Encoder/batch_normalization_56/ReadVariableOp_12T
(Encoder/conv2d_37/BiasAdd/ReadVariableOp(Encoder/conv2d_37/BiasAdd/ReadVariableOp2R
'Encoder/conv2d_37/Conv2D/ReadVariableOp'Encoder/conv2d_37/Conv2D/ReadVariableOp2T
(Encoder/conv2d_38/BiasAdd/ReadVariableOp(Encoder/conv2d_38/BiasAdd/ReadVariableOp2R
'Encoder/conv2d_38/Conv2D/ReadVariableOp'Encoder/conv2d_38/Conv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

G
+__inference_lambda_27_layer_call_fn_8188498

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_27_layer_call_and_return_conditional_losses_8186780z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С
у
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187354
input_59*
encoder_8187303:
encoder_8187305:	
encoder_8187307:	
encoder_8187309:	
encoder_8187311:	
encoder_8187313:	*
encoder_8187315:
encoder_8187317:
encoder_8187319:
encoder_8187321:
encoder_8187323:
encoder_8187325:*
decoder_8187328:
decoder_8187330:	
decoder_8187332:	
decoder_8187334:	
decoder_8187336:	
decoder_8187338:	*
decoder_8187340:
decoder_8187342:
decoder_8187344:
decoder_8187346:
decoder_8187348:
decoder_8187350:
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallК
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_59encoder_8187303encoder_8187305encoder_8187307encoder_8187309encoder_8187311encoder_8187313encoder_8187315encoder_8187317encoder_8187319encoder_8187321encoder_8187323encoder_8187325*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186144ь
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_8187328decoder_8187330decoder_8187332decoder_8187334decoder_8187336decoder_8187338decoder_8187340decoder_8187342decoder_8187344decoder_8187346decoder_8187348decoder_8187350*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_8186722
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59
	
з
8__inference_batch_normalization_57_layer_call_fn_8188595

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186513
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л"
 
D__inference_Encoder_layer_call_and_return_conditional_losses_8186144

inputs,
conv2d_37_8186101: 
conv2d_37_8186103:	-
batch_normalization_55_8186107:	-
batch_normalization_55_8186109:	-
batch_normalization_55_8186111:	-
batch_normalization_55_8186113:	,
conv2d_38_8186128:
conv2d_38_8186130:,
batch_normalization_56_8186134:,
batch_normalization_56_8186136:,
batch_normalization_56_8186138:,
batch_normalization_56_8186140:
identityЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ!conv2d_37/StatefulPartitionedCallЂ!conv2d_38/StatefulPartitionedCall
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_37_8186101conv2d_37_8186103*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100љ
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_55_8186107batch_normalization_55_8186109batch_normalization_55_8186111batch_normalization_55_8186113*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185964Г
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0conv2d_38_8186128conv2d_38_8186130*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127ј
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_56_8186134batch_normalization_56_8186136batch_normalization_56_8186138batch_normalization_56_8186140*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186040
IdentityIdentity7batch_normalization_56/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ№
NoOpNoOp/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ј*

P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8188729

inputsC
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :U
subSubstrided_slice_1:output:0sub/y:output:0*
T0*
_output_shapes
: G
mul/yConst*
_output_shapes
: *
dtype0*
value	B :D
mulMulsub:z:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : H
sub_1Subadd:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	sub_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_2AddV2	mul_1:z:0add_2/y:output:0*
T0*
_output_shapes
: I
sub_3/yConst*
_output_shapes
: *
dtype0*
value	B : J
sub_3Sub	add_2:z:0sub_3/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_3AddV2	sub_3:z:0add_3/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :{
stackPackstrided_slice:output:0	add_1:z:0	add_3:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 	
з
8__inference_batch_normalization_55_layer_call_fn_8188330

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185964
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188631

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185995

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
з"
 
D__inference_Encoder_layer_call_and_return_conditional_losses_8186256

inputs,
conv2d_37_8186225: 
conv2d_37_8186227:	-
batch_normalization_55_8186231:	-
batch_normalization_55_8186233:	-
batch_normalization_55_8186235:	-
batch_normalization_55_8186237:	,
conv2d_38_8186240:
conv2d_38_8186242:,
batch_normalization_56_8186246:,
batch_normalization_56_8186248:,
batch_normalization_56_8186250:,
batch_normalization_56_8186252:
identityЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ!conv2d_37/StatefulPartitionedCallЂ!conv2d_38/StatefulPartitionedCall
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_37_8186225conv2d_37_8186227*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100љ
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_55_8186231batch_normalization_55_8186233batch_normalization_55_8186235batch_normalization_55_8186237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185995Г
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0conv2d_38_8186240conv2d_38_8186242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8186127ј
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8186015
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_56_8186246batch_normalization_56_8186248batch_normalization_56_8186250batch_normalization_56_8186252*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8186071
IdentityIdentity7batch_normalization_56/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ№
NoOpNoOp/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_8187897
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable


F__inference_conv2d_37_layer_call_and_return_conditional_losses_8188307

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
№

=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187571

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:%

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187196
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ф
b
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186763

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               v
PadPadinputsPad/paddings:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџo
IdentityIdentityPad:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
b
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188504

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               u
PadPadinputsPad/paddings:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџn
IdentityIdentityPad:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
J
к
D__inference_Encoder_layer_call_and_return_conditional_losses_8188061

inputsC
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	=
.batch_normalization_55_readvariableop_resource:	?
0batch_normalization_55_readvariableop_1_resource:	N
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_38_conv2d_readvariableop_resource:7
)conv2d_38_biasadd_readvariableop_resource:<
.batch_normalization_56_readvariableop_resource:>
0batch_normalization_56_readvariableop_1_resource:M
?batch_normalization_56_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:
identityЂ%batch_normalization_55/AssignNewValueЂ'batch_normalization_55/AssignNewValue_1Ђ6batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_55/ReadVariableOpЂ'batch_normalization_55/ReadVariableOp_1Ђ%batch_normalization_56/AssignNewValueЂ'batch_normalization_56/AssignNewValue_1Ђ6batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_56/ReadVariableOpЂ'batch_normalization_56/ReadVariableOp_1Ђ conv2d_37/BiasAdd/ReadVariableOpЂconv2d_37/Conv2D/ReadVariableOpЂ conv2d_38/BiasAdd/ReadVariableOpЂconv2d_38/Conv2D/ReadVariableOp
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Џ
conv2d_37/Conv2DConv2Dinputs'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides

 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?k
conv2d_37/EluEluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?Ў
max_pooling2d_33/MaxPoolMaxPoolconv2d_37/Elu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0г
conv2d_38/Conv2DConv2D+batch_normalization_55/FusedBatchNormV3:y:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_38/EluEluconv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ­
max_pooling2d_34/MaxPoolMaxPoolconv2d_38/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0в
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_56/AssignNewValueAssignVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource4batch_normalization_56/FusedBatchNormV3:batch_mean:07^batch_normalization_56/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_56/AssignNewValue_1AssignVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_56/FusedBatchNormV3:batch_variance:09^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_56/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
NoOpNoOp&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1&^batch_normalization_56/AssignNewValue(^batch_normalization_56/AssignNewValue_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_1!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 2N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_56/AssignNewValue%batch_normalization_56/AssignNewValue2R
'batch_normalization_56/AssignNewValue_1'batch_normalization_56/AssignNewValue_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д
Y
$__inference__update_step_xla_8187892
gradient#
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:: *
	_noinline(:Q M
'
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ф
b
F__inference_lambda_28_layer_call_and_return_conditional_losses_8186705

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               v
PadPadinputsPad/paddings:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџo
IdentityIdentityPad:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
д
Y
$__inference__update_step_xla_8187832
gradient#
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:: *
	_noinline(:Q M
'
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
з
8__inference_batch_normalization_55_layer_call_fn_8188343

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8185995
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЭЕ

X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187699

inputsK
0encoder_conv2d_37_conv2d_readvariableop_resource:@
1encoder_conv2d_37_biasadd_readvariableop_resource:	E
6encoder_batch_normalization_55_readvariableop_resource:	G
8encoder_batch_normalization_55_readvariableop_1_resource:	V
Gencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:	X
Iencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:	K
0encoder_conv2d_38_conv2d_readvariableop_resource:?
1encoder_conv2d_38_biasadd_readvariableop_resource:D
6encoder_batch_normalization_56_readvariableop_resource:F
8encoder_batch_normalization_56_readvariableop_1_resource:U
Gencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:_
Ddecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource:J
;decoder_conv2d_transpose_27_biasadd_readvariableop_resource:	E
6decoder_batch_normalization_57_readvariableop_resource:	G
8decoder_batch_normalization_57_readvariableop_1_resource:	V
Gdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource:	X
Idecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:	_
Ddecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_28_biasadd_readvariableop_resource:D
6decoder_batch_normalization_58_readvariableop_resource:F
8decoder_batch_normalization_58_readvariableop_1_resource:U
Gdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource:W
Idecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource:
identityЂ>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ-Decoder/batch_normalization_57/ReadVariableOpЂ/Decoder/batch_normalization_57/ReadVariableOp_1Ђ>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ-Decoder/batch_normalization_58/ReadVariableOpЂ/Decoder/batch_normalization_58/ReadVariableOp_1Ђ2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpЂ;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpЂ;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂ>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ-Encoder/batch_normalization_55/ReadVariableOpЂ/Encoder/batch_normalization_55/ReadVariableOp_1Ђ>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ-Encoder/batch_normalization_56/ReadVariableOpЂ/Encoder/batch_normalization_56/ReadVariableOp_1Ђ(Encoder/conv2d_37/BiasAdd/ReadVariableOpЂ'Encoder/conv2d_37/Conv2D/ReadVariableOpЂ(Encoder/conv2d_38/BiasAdd/ReadVariableOpЂ'Encoder/conv2d_38/Conv2D/ReadVariableOpЁ
'Encoder/conv2d_37/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0П
Encoder/conv2d_37/Conv2DConv2Dinputs/Encoder/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides

(Encoder/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
Encoder/conv2d_37/BiasAddBiasAdd!Encoder/conv2d_37/Conv2D:output:00Encoder/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?{
Encoder/conv2d_37/EluElu"Encoder/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?О
 Encoder/max_pooling2d_33/MaxPoolMaxPool#Encoder/conv2d_37/Elu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Ё
-Encoder/batch_normalization_55/ReadVariableOpReadVariableOp6encoder_batch_normalization_55_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/Encoder/batch_normalization_55/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_55_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0љ
/Encoder/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3)Encoder/max_pooling2d_33/MaxPool:output:05Encoder/batch_normalization_55/ReadVariableOp:value:07Encoder/batch_normalization_55/ReadVariableOp_1:value:0FEncoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0HEncoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ё
'Encoder/conv2d_38/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ы
Encoder/conv2d_38/Conv2DConv2D3Encoder/batch_normalization_55/FusedBatchNormV3:y:0/Encoder/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

(Encoder/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
Encoder/conv2d_38/BiasAddBiasAdd!Encoder/conv2d_38/Conv2D:output:00Encoder/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџz
Encoder/conv2d_38/EluElu"Encoder/conv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџН
 Encoder/max_pooling2d_34/MaxPoolMaxPool#Encoder/conv2d_38/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
 
-Encoder/batch_normalization_56/ReadVariableOpReadVariableOp6encoder_batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype0Є
/Encoder/batch_normalization_56/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0є
/Encoder/batch_normalization_56/FusedBatchNormV3FusedBatchNormV3)Encoder/max_pooling2d_34/MaxPool:output:05Encoder/batch_normalization_56/ReadVariableOp:value:07Encoder/batch_normalization_56/ReadVariableOp_1:value:0FEncoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0HEncoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( o
Decoder/up_sampling2d_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 Decoder/up_sampling2d_27/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_27/mulMul'Decoder/up_sampling2d_27/Const:output:0)Decoder/up_sampling2d_27/Const_1:output:0*
T0*
_output_shapes
:љ
5Decoder/up_sampling2d_27/resize/ResizeNearestNeighborResizeNearestNeighbor3Encoder/batch_normalization_56/FusedBatchNormV3:y:0 Decoder/up_sampling2d_27/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(
Decoder/lambda_27/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Ч
Decoder/lambda_27/PadPadFDecoder/up_sampling2d_27/resize/ResizeNearestNeighbor:resized_images:0'Decoder/lambda_27/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџo
!Decoder/conv2d_transpose_27/ShapeShapeDecoder/lambda_27/Pad:output:0*
T0*
_output_shapes
:y
/Decoder/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1Decoder/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1Decoder/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)Decoder/conv2d_transpose_27/strided_sliceStridedSlice*Decoder/conv2d_transpose_27/Shape:output:08Decoder/conv2d_transpose_27/strided_slice/stack:output:0:Decoder/conv2d_transpose_27/strided_slice/stack_1:output:0:Decoder/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#Decoder/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#Decoder/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#Decoder/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!Decoder/conv2d_transpose_27/stackPack2Decoder/conv2d_transpose_27/strided_slice:output:0,Decoder/conv2d_transpose_27/stack/1:output:0,Decoder/conv2d_transpose_27/stack/2:output:0,Decoder/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:{
1Decoder/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Decoder/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Decoder/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+Decoder/conv2d_transpose_27/strided_slice_1StridedSlice*Decoder/conv2d_transpose_27/stack:output:0:Decoder/conv2d_transpose_27/strided_slice_1/stack:output:0<Decoder/conv2d_transpose_27/strided_slice_1/stack_1:output:0<Decoder/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0И
,Decoder/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput*Decoder/conv2d_transpose_27/stack:output:0CDecoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0Decoder/lambda_27/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
Ћ
2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#Decoder/conv2d_transpose_27/BiasAddBiasAdd5Decoder/conv2d_transpose_27/conv2d_transpose:output:0:Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
Decoder/conv2d_transpose_27/EluElu,Decoder/conv2d_transpose_27/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЁ
-Decoder/batch_normalization_57/ReadVariableOpReadVariableOp6decoder_batch_normalization_57_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/Decoder/batch_normalization_57/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0§
/Decoder/batch_normalization_57/FusedBatchNormV3FusedBatchNormV3-Decoder/conv2d_transpose_27/Elu:activations:05Decoder/batch_normalization_57/ReadVariableOp:value:07Decoder/batch_normalization_57/ReadVariableOp_1:value:0FDecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0HDecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( o
Decoder/up_sampling2d_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 Decoder/up_sampling2d_28/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_28/mulMul'Decoder/up_sampling2d_28/Const:output:0)Decoder/up_sampling2d_28/Const_1:output:0*
T0*
_output_shapes
:њ
5Decoder/up_sampling2d_28/resize/ResizeNearestNeighborResizeNearestNeighbor3Decoder/batch_normalization_57/FusedBatchNormV3:y:0 Decoder/up_sampling2d_28/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ>*
half_pixel_centers(
Decoder/lambda_28/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Ш
Decoder/lambda_28/PadPadFDecoder/up_sampling2d_28/resize/ResizeNearestNeighbor:resized_images:0'Decoder/lambda_28/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?o
!Decoder/conv2d_transpose_28/ShapeShapeDecoder/lambda_28/Pad:output:0*
T0*
_output_shapes
:y
/Decoder/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1Decoder/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1Decoder/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)Decoder/conv2d_transpose_28/strided_sliceStridedSlice*Decoder/conv2d_transpose_28/Shape:output:08Decoder/conv2d_transpose_28/strided_slice/stack:output:0:Decoder/conv2d_transpose_28/strided_slice/stack_1:output:0:Decoder/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#Decoder/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@f
#Decoder/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value
B :e
#Decoder/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
!Decoder/conv2d_transpose_28/stackPack2Decoder/conv2d_transpose_28/strided_slice:output:0,Decoder/conv2d_transpose_28/stack/1:output:0,Decoder/conv2d_transpose_28/stack/2:output:0,Decoder/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:{
1Decoder/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Decoder/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Decoder/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+Decoder/conv2d_transpose_28/strided_slice_1StridedSlice*Decoder/conv2d_transpose_28/stack:output:0:Decoder/conv2d_transpose_28/strided_slice_1/stack:output:0<Decoder/conv2d_transpose_28/strided_slice_1/stack_1:output:0<Decoder/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype0И
,Decoder/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput*Decoder/conv2d_transpose_28/stack:output:0CDecoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0Decoder/lambda_28/Pad:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
Њ
2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
#Decoder/conv2d_transpose_28/BiasAddBiasAdd5Decoder/conv2d_transpose_28/conv2d_transpose:output:0:Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@
Decoder/conv2d_transpose_28/EluElu,Decoder/conv2d_transpose_28/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@ 
-Decoder/batch_normalization_58/ReadVariableOpReadVariableOp6decoder_batch_normalization_58_readvariableop_resource*
_output_shapes
:*
dtype0Є
/Decoder/batch_normalization_58/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_58_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0љ
/Decoder/batch_normalization_58/FusedBatchNormV3FusedBatchNormV3-Decoder/conv2d_transpose_28/Elu:activations:05Decoder/batch_normalization_58/ReadVariableOp:value:07Decoder/batch_normalization_58/ReadVariableOp_1:value:0FDecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0HDecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ@:::::*
epsilon%o:*
is_training( 
IdentityIdentity3Decoder/batch_normalization_58/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ю

NoOpNoOp?^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpA^Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1.^Decoder/batch_normalization_57/ReadVariableOp0^Decoder/batch_normalization_57/ReadVariableOp_1?^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpA^Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1.^Decoder/batch_normalization_58/ReadVariableOp0^Decoder/batch_normalization_58/ReadVariableOp_13^Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp<^Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp3^Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp<^Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp?^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpA^Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1.^Encoder/batch_normalization_55/ReadVariableOp0^Encoder/batch_normalization_55/ReadVariableOp_1?^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpA^Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1.^Encoder/batch_normalization_56/ReadVariableOp0^Encoder/batch_normalization_56/ReadVariableOp_1)^Encoder/conv2d_37/BiasAdd/ReadVariableOp(^Encoder/conv2d_37/Conv2D/ReadVariableOp)^Encoder/conv2d_38/BiasAdd/ReadVariableOp(^Encoder/conv2d_38/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 2
>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp>Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp2
@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1@Decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12^
-Decoder/batch_normalization_57/ReadVariableOp-Decoder/batch_normalization_57/ReadVariableOp2b
/Decoder/batch_normalization_57/ReadVariableOp_1/Decoder/batch_normalization_57/ReadVariableOp_12
>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp>Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp2
@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1@Decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12^
-Decoder/batch_normalization_58/ReadVariableOp-Decoder/batch_normalization_58/ReadVariableOp2b
/Decoder/batch_normalization_58/ReadVariableOp_1/Decoder/batch_normalization_58/ReadVariableOp_12h
2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2Decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2z
;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp;Decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2h
2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2Decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2z
;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp;Decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2
>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp>Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2
@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1@Encoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12^
-Encoder/batch_normalization_55/ReadVariableOp-Encoder/batch_normalization_55/ReadVariableOp2b
/Encoder/batch_normalization_55/ReadVariableOp_1/Encoder/batch_normalization_55/ReadVariableOp_12
>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp>Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp2
@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1@Encoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12^
-Encoder/batch_normalization_56/ReadVariableOp-Encoder/batch_normalization_56/ReadVariableOp2b
/Encoder/batch_normalization_56/ReadVariableOp_1/Encoder/batch_normalization_56/ReadVariableOp_12T
(Encoder/conv2d_37/BiasAdd/ReadVariableOp(Encoder/conv2d_37/BiasAdd/ReadVariableOp2R
'Encoder/conv2d_37/Conv2D/ReadVariableOp'Encoder/conv2d_37/Conv2D/ReadVariableOp2T
(Encoder/conv2d_38/BiasAdd/ReadVariableOp(Encoder/conv2d_38/BiasAdd/ReadVariableOp2R
'Encoder/conv2d_38/Conv2D/ReadVariableOp'Encoder/conv2d_38/Conv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187837
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

Ќ
)__inference_Encoder_layer_call_fn_8187936

inputs"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186144w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8188648

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187300
input_59"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:%

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinput_59unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187196
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59
 	
з
8__inference_batch_normalization_57_layer_call_fn_8188582

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8186482
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188379

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_8187847
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ў

=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187087
input_59"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:%

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_59unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187036
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:џџџџџџџџџ@: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_59


F__inference_conv2d_37_layer_call_and_return_conditional_losses_8186100

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ?W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ?i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ў
)__inference_Encoder_layer_call_fn_8186171
input_57"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	$
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_57unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_8186144w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_57
д
Y
$__inference__update_step_xla_8187852
gradient#
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:: *
	_noinline(:Q M
'
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
П
N
2__inference_max_pooling2d_33_layer_call_fn_8188312

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8185939
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
F
input_59:
serving_default_input_59:0џџџџџџџџџ@D
Decoder9
StatefulPartitionedCall:0џџџџџџџџџ@tensorflow/serving/predict:К
у
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_sequential

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
 layer-4
!layer-5
"layer_with_weights-2
"layer-6
#layer_with_weights-3
#layer-7
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
#*_self_saveable_object_factories"
_tf_keras_sequential
ж
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23"
trackable_list_wrapper

+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Љ
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32О
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187087
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187518
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187571
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187300П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3

Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32Њ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187699
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187827
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187354
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187408П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
ЮBЫ
"__inference__wrapped_model_8185930input_59"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper

X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

+kernel
,bias
#^_self_saveable_object_factories
 __jit_compiled_convolution_op"
_tf_keras_layer
Ъ
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
#f_self_saveable_object_factories"
_tf_keras_layer

g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	-gamma
.beta
/moving_mean
0moving_variance
#n_self_saveable_object_factories"
_tf_keras_layer

o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

1kernel
2bias
#u_self_saveable_object_factories
 v_jit_compiled_convolution_op"
_tf_keras_layer
Ъ
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
#}_self_saveable_object_factories"
_tf_keras_layer

~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	3gamma
4beta
5moving_mean
6moving_variance
$_self_saveable_object_factories"
_tf_keras_layer
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611"
trackable_list_wrapper
X
+0
,1
-2
.3
14
25
36
47"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
trace_0
trace_1
trace_2
trace_32ю
)__inference_Encoder_layer_call_fn_8186171
)__inference_Encoder_layer_call_fn_8187936
)__inference_Encoder_layer_call_fn_8187965
)__inference_Encoder_layer_call_fn_8186312П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Э
trace_0
trace_1
trace_2
trace_32к
D__inference_Encoder_layer_call_and_return_conditional_losses_8188013
D__inference_Encoder_layer_call_and_return_conditional_losses_8188061
D__inference_Encoder_layer_call_and_return_conditional_losses_8186346
D__inference_Encoder_layer_call_and_return_conditional_losses_8186380П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
 "
trackable_dict_wrapper
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories"
_tf_keras_layer
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$ _self_saveable_object_factories"
_tf_keras_layer

Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses

7kernel
8bias
$Ї_self_saveable_object_factories
!Ј_jit_compiled_convolution_op"
_tf_keras_layer

Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
	Џaxis
	9gamma
:beta
;moving_mean
<moving_variance
$А_self_saveable_object_factories"
_tf_keras_layer
б
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
$З_self_saveable_object_factories"
_tf_keras_layer
б
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses
$О_self_saveable_object_factories"
_tf_keras_layer

П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses

=kernel
>bias
$Х_self_saveable_object_factories
!Ц_jit_compiled_convolution_op"
_tf_keras_layer

Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
	Эaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance
$Ю_self_saveable_object_factories"
_tf_keras_layer
v
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11"
trackable_list_wrapper
X
70
81
92
:3
=4
>5
?6
@7"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
с
дtrace_0
еtrace_1
жtrace_2
зtrace_32ю
)__inference_Decoder_layer_call_fn_8186749
)__inference_Decoder_layer_call_fn_8188090
)__inference_Decoder_layer_call_fn_8188119
)__inference_Decoder_layer_call_fn_8186906П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0zеtrace_1zжtrace_2zзtrace_3
Э
иtrace_0
йtrace_1
кtrace_2
лtrace_32к
D__inference_Decoder_layer_call_and_return_conditional_losses_8188203
D__inference_Decoder_layer_call_and_return_conditional_losses_8188287
D__inference_Decoder_layer_call_and_return_conditional_losses_8186942
D__inference_Decoder_layer_call_and_return_conditional_losses_8186978П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0zйtrace_1zкtrace_2zлtrace_3
 "
trackable_dict_wrapper
+:)2conv2d_37/kernel
:2conv2d_37/bias
+:)2batch_normalization_55/gamma
*:(2batch_normalization_55/beta
3:1 (2"batch_normalization_55/moving_mean
7:5 (2&batch_normalization_55/moving_variance
+:)2conv2d_38/kernel
:2conv2d_38/bias
*:(2batch_normalization_56/gamma
):'2batch_normalization_56/beta
2:0 (2"batch_normalization_56/moving_mean
6:4 (2&batch_normalization_56/moving_variance
5:32conv2d_transpose_27/kernel
':%2conv2d_transpose_27/bias
+:)2batch_normalization_57/gamma
*:(2batch_normalization_57/beta
3:1 (2"batch_normalization_57/moving_mean
7:5 (2&batch_normalization_57/moving_variance
5:32conv2d_transpose_28/kernel
&:$2conv2d_transpose_28/bias
*:(2batch_normalization_58/gamma
):'2batch_normalization_58/beta
2:0 (2"batch_normalization_58/moving_mean
6:4 (2&batch_normalization_58/moving_variance
X
/0
01
52
63
;4
<5
A6
B7"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
м0
н1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187087input_59"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187518inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187571inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187300input_59"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЉBІ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187699inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЉBІ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187827inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187354input_59"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187408input_59"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О
Q0
о1
п2
р3
с4
т5
у6
ф7
х8
ц9
ч10
ш11
щ12
ъ13
ы14
ь15
э16
ю17
я18
№19
ё20
ђ21
ѓ22
є23
ѕ24
і25
ї26
ј27
љ28
њ29
ћ30
ќ31
§32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
І
о0
р1
т2
ф3
ц4
ш5
ъ6
ь7
ю8
№9
ђ10
є11
і12
ј13
њ14
ќ15"
trackable_list_wrapper
І
п0
с1
у2
х3
ч4
щ5
ы6
э7
я8
ё9
ѓ10
ѕ11
ї12
љ13
ћ14
§15"
trackable_list_wrapper
ы	
ўtrace_0
џtrace_1
trace_2
trace_3
trace_4
trace_5
trace_6
trace_7
trace_8
trace_9
trace_10
trace_11
trace_12
trace_13
trace_14
trace_152
$__inference__update_step_xla_8187832
$__inference__update_step_xla_8187837
$__inference__update_step_xla_8187842
$__inference__update_step_xla_8187847
$__inference__update_step_xla_8187852
$__inference__update_step_xla_8187857
$__inference__update_step_xla_8187862
$__inference__update_step_xla_8187867
$__inference__update_step_xla_8187872
$__inference__update_step_xla_8187877
$__inference__update_step_xla_8187882
$__inference__update_step_xla_8187887
$__inference__update_step_xla_8187892
$__inference__update_step_xla_8187897
$__inference__update_step_xla_8187902
$__inference__update_step_xla_8187907Й
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zўtrace_0zџtrace_1ztrace_2ztrace_3ztrace_4ztrace_5ztrace_6ztrace_7ztrace_8ztrace_9ztrace_10ztrace_11ztrace_12ztrace_13ztrace_14ztrace_15
ЭBЪ
%__inference_signature_wrapper_8187465input_59"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_conv2d_37_layer_call_fn_8188296Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8188307Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
ј
trace_02й
2__inference_max_pooling2d_33_layer_call_fn_8188312Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02є
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8188317Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
х
Ёtrace_0
Ђtrace_12Њ
8__inference_batch_normalization_55_layer_call_fn_8188330
8__inference_batch_normalization_55_layer_call_fn_8188343Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0zЂtrace_1

Ѓtrace_0
Єtrace_12р
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188361
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188379Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0zЄtrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ё
Њtrace_02в
+__inference_conv2d_38_layer_call_fn_8188388Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0

Ћtrace_02э
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8188399Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЋtrace_0
 "
trackable_dict_wrapper
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ј
Бtrace_02й
2__inference_max_pooling2d_34_layer_call_fn_8188404Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0

Вtrace_02є
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8188409Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zВtrace_0
 "
trackable_dict_wrapper
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
Иtrace_0
Йtrace_12Њ
8__inference_batch_normalization_56_layer_call_fn_8188422
8__inference_batch_normalization_56_layer_call_fn_8188435Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0zЙtrace_1

Кtrace_0
Лtrace_12р
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188453
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188471Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zКtrace_0zЛtrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
/0
01
52
63"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
)__inference_Encoder_layer_call_fn_8186171input_57"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
)__inference_Encoder_layer_call_fn_8187936inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
)__inference_Encoder_layer_call_fn_8187965inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
)__inference_Encoder_layer_call_fn_8186312input_57"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_8188013inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_8188061inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_8186346input_57"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_8186380input_57"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ј
Сtrace_02й
2__inference_up_sampling2d_27_layer_call_fn_8188476Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zСtrace_0

Тtrace_02є
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8188488Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
з
Шtrace_0
Щtrace_12
+__inference_lambda_27_layer_call_fn_8188493
+__inference_lambda_27_layer_call_fn_8188498П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zШtrace_0zЩtrace_1

Ъtrace_0
Ыtrace_12в
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188504
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188510П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0zЫtrace_1
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
ћ
бtrace_02м
5__inference_conv2d_transpose_27_layer_call_fn_8188519Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0

вtrace_02ї
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8188569Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0
 "
trackable_dict_wrapper
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
х
иtrace_0
йtrace_12Њ
8__inference_batch_normalization_57_layer_call_fn_8188582
8__inference_batch_normalization_57_layer_call_fn_8188595Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0zйtrace_1

кtrace_0
лtrace_12р
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188613
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188631Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0zлtrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
ј
сtrace_02й
2__inference_up_sampling2d_28_layer_call_fn_8188636Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zсtrace_0

тtrace_02є
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8188648Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
з
шtrace_0
щtrace_12
+__inference_lambda_28_layer_call_fn_8188653
+__inference_lambda_28_layer_call_fn_8188658П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0zщtrace_1

ъtrace_0
ыtrace_12в
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188664
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188670П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0zыtrace_1
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
ћ
ёtrace_02м
5__inference_conv2d_transpose_28_layer_call_fn_8188679Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0

ђtrace_02ї
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8188729Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zђtrace_0
 "
trackable_dict_wrapper
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
х
јtrace_0
љtrace_12Њ
8__inference_batch_normalization_58_layer_call_fn_8188742
8__inference_batch_normalization_58_layer_call_fn_8188755Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0zљtrace_1

њtrace_0
ћtrace_12р
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188773
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188791Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zњtrace_0zћtrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
;0
<1
A2
B3"
trackable_list_wrapper
X
0
1
2
3
 4
!5
"6
#7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
)__inference_Decoder_layer_call_fn_8186749input_58"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
)__inference_Decoder_layer_call_fn_8188090inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
)__inference_Decoder_layer_call_fn_8188119inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
)__inference_Decoder_layer_call_fn_8186906input_58"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_8188203inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_8188287inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_8186942input_58"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_8186978input_58"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
ќ	variables
§	keras_api

ўtotal

џcount"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0:.2Adam/m/conv2d_37/kernel
0:.2Adam/v/conv2d_37/kernel
": 2Adam/m/conv2d_37/bias
": 2Adam/v/conv2d_37/bias
0:.2#Adam/m/batch_normalization_55/gamma
0:.2#Adam/v/batch_normalization_55/gamma
/:-2"Adam/m/batch_normalization_55/beta
/:-2"Adam/v/batch_normalization_55/beta
0:.2Adam/m/conv2d_38/kernel
0:.2Adam/v/conv2d_38/kernel
!:2Adam/m/conv2d_38/bias
!:2Adam/v/conv2d_38/bias
/:-2#Adam/m/batch_normalization_56/gamma
/:-2#Adam/v/batch_normalization_56/gamma
.:,2"Adam/m/batch_normalization_56/beta
.:,2"Adam/v/batch_normalization_56/beta
::82!Adam/m/conv2d_transpose_27/kernel
::82!Adam/v/conv2d_transpose_27/kernel
,:*2Adam/m/conv2d_transpose_27/bias
,:*2Adam/v/conv2d_transpose_27/bias
0:.2#Adam/m/batch_normalization_57/gamma
0:.2#Adam/v/batch_normalization_57/gamma
/:-2"Adam/m/batch_normalization_57/beta
/:-2"Adam/v/batch_normalization_57/beta
::82!Adam/m/conv2d_transpose_28/kernel
::82!Adam/v/conv2d_transpose_28/kernel
+:)2Adam/m/conv2d_transpose_28/bias
+:)2Adam/v/conv2d_transpose_28/bias
/:-2#Adam/m/batch_normalization_58/gamma
/:-2#Adam/v/batch_normalization_58/gamma
.:,2"Adam/m/batch_normalization_58/beta
.:,2"Adam/v/batch_normalization_58/beta
љBі
$__inference__update_step_xla_8187832gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187837gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187842gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187847gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187852gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187857gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187862gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187867gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187872gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187877gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187882gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187887gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187892gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187897gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187902gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_8187907gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_conv2d_37_layer_call_fn_8188296inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8188307inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
2__inference_max_pooling2d_33_layer_call_fn_8188312inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8188317inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
8__inference_batch_normalization_55_layer_call_fn_8188330inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
8__inference_batch_normalization_55_layer_call_fn_8188343inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188361inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188379inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_conv2d_38_layer_call_fn_8188388inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8188399inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
2__inference_max_pooling2d_34_layer_call_fn_8188404inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8188409inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
8__inference_batch_normalization_56_layer_call_fn_8188422inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
8__inference_batch_normalization_56_layer_call_fn_8188435inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188453inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_8188471inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
2__inference_up_sampling2d_27_layer_call_fn_8188476inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8188488inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
+__inference_lambda_27_layer_call_fn_8188493inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
+__inference_lambda_27_layer_call_fn_8188498inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188504inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188510inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_conv2d_transpose_27_layer_call_fn_8188519inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_8188569inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
8__inference_batch_normalization_57_layer_call_fn_8188582inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
8__inference_batch_normalization_57_layer_call_fn_8188595inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188613inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_8188631inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
2__inference_up_sampling2d_28_layer_call_fn_8188636inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8188648inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
+__inference_lambda_28_layer_call_fn_8188653inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
+__inference_lambda_28_layer_call_fn_8188658inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188664inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188670inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_conv2d_transpose_28_layer_call_fn_8188679inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8188729inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
8__inference_batch_normalization_58_layer_call_fn_8188742inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
8__inference_batch_normalization_58_layer_call_fn_8188755inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188773inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188791inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
ў0
џ1"
trackable_list_wrapper
.
ќ	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperт
D__inference_Decoder_layer_call_and_return_conditional_losses_8186942789:;<=>?@ABAЂ>
7Ђ4
*'
input_58џџџџџџџџџ
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 т
D__inference_Decoder_layer_call_and_return_conditional_losses_8186978789:;<=>?@ABAЂ>
7Ђ4
*'
input_58џџџџџџџџџ
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
D__inference_Decoder_layer_call_and_return_conditional_losses_8188203789:;<=>?@AB?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 Я
D__inference_Decoder_layer_call_and_return_conditional_losses_8188287789:;<=>?@AB?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 М
)__inference_Decoder_layer_call_fn_8186749789:;<=>?@ABAЂ>
7Ђ4
*'
input_58џџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
)__inference_Decoder_layer_call_fn_8186906789:;<=>?@ABAЂ>
7Ђ4
*'
input_58џџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
)__inference_Decoder_layer_call_fn_8188090789:;<=>?@AB?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
)__inference_Decoder_layer_call_fn_8188119789:;<=>?@AB?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџб
D__inference_Encoder_layer_call_and_return_conditional_losses_8186346+,-./0123456BЂ?
8Ђ5
+(
input_57џџџџџџџџџ@
p 

 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 б
D__inference_Encoder_layer_call_and_return_conditional_losses_8186380+,-./0123456BЂ?
8Ђ5
+(
input_57џџџџџџџџџ@
p

 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 Я
D__inference_Encoder_layer_call_and_return_conditional_losses_8188013+,-./0123456@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 Я
D__inference_Encoder_layer_call_and_return_conditional_losses_8188061+,-./0123456@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 Њ
)__inference_Encoder_layer_call_fn_8186171}+,-./0123456BЂ?
8Ђ5
+(
input_57џџџџџџџџџ@
p 

 
Њ ")&
unknownџџџџџџџџџЊ
)__inference_Encoder_layer_call_fn_8186312}+,-./0123456BЂ?
8Ђ5
+(
input_57џџџџџџџџџ@
p

 
Њ ")&
unknownџџџџџџџџџЈ
)__inference_Encoder_layer_call_fn_8187936{+,-./0123456@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ ")&
unknownџџџџџџџџџЈ
)__inference_Encoder_layer_call_fn_8187965{+,-./0123456@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ ")&
unknownџџџџџџџџџ
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187354І+,-./0123456789:;<=>?@ABBЂ?
8Ђ5
+(
input_59џџџџџџџџџ@
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187408І+,-./0123456789:;<=>?@ABBЂ?
8Ђ5
+(
input_59џџџџџџџџџ@
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 №
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187699+,-./0123456789:;<=>?@AB@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 №
X__inference_Fully_Convolutional_AE_STFT_layer_call_and_return_conditional_losses_8187827+,-./0123456789:;<=>?@AB@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 н
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187087+,-./0123456789:;<=>?@ABBЂ?
8Ђ5
+(
input_59џџџџџџџџџ@
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџн
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187300+,-./0123456789:;<=>?@ABBЂ?
8Ђ5
+(
input_59џџџџџџџџџ@
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџл
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187518+,-./0123456789:;<=>?@AB@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџл
=__inference_Fully_Convolutional_AE_STFT_layer_call_fn_8187571+,-./0123456789:;<=>?@AB@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
$__inference__update_step_xla_8187832zЂw
pЂm
"
gradient
=:	&Ђ#
њ

p
` VariableSpec 
`рыЦцв?
Њ "
 
$__inference__update_step_xla_8187837hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`рЏЕВЁв?
Њ "
 
$__inference__update_step_xla_8187842hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`роАав?
Њ "
 
$__inference__update_step_xla_8187847hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`рЫўв?
Њ "
 Љ
$__inference__update_step_xla_8187852zЂw
pЂm
"
gradient
=:	&Ђ#
њ

p
` VariableSpec 
`рбїв?
Њ "
 
$__inference__update_step_xla_8187857f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рпУўв?
Њ "
 
$__inference__update_step_xla_8187862f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЅЪв?
Њ "
 
$__inference__update_step_xla_8187867f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рївФв?
Њ "
 Љ
$__inference__update_step_xla_8187872zЂw
pЂm
"
gradient
=:	&Ђ#
њ

p
` VariableSpec 
`рВЁэв?
Њ "
 
$__inference__update_step_xla_8187877hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`рЩШв?
Њ "
 
$__inference__update_step_xla_8187882hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`рВН№в?
Њ "
 
$__inference__update_step_xla_8187887hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`рЕН№в?
Њ "
 Љ
$__inference__update_step_xla_8187892zЂw
pЂm
"
gradient
=:	&Ђ#
њ

p
` VariableSpec 
`рлШв?
Њ "
 
$__inference__update_step_xla_8187897f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЈВЁв?
Њ "
 
$__inference__update_step_xla_8187902f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рљЉВЁв?
Њ "
 
$__inference__update_step_xla_8187907f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЙЈВЁв?
Њ "
 Й
"__inference__wrapped_model_8185930+,-./0123456789:;<=>?@AB:Ђ7
0Ђ-
+(
input_59џџџџџџџџџ@
Њ ":Њ7
5
Decoder*'
decoderџџџџџџџџџ@ї
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188361-./0NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ї
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_8188379-./0NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
8__inference_batch_normalization_55_layer_call_fn_8188330-./0NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџб
8__inference_batch_normalization_55_layer_call_fn_8188343-./0NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџѕ
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_81884533456MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ѕ
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_81884713456MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
8__inference_batch_normalization_56_layer_call_fn_81884223456MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЯ
8__inference_batch_normalization_56_layer_call_fn_81884353456MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџї
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_81886139:;<NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ї
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_81886319:;<NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
8__inference_batch_normalization_57_layer_call_fn_81885829:;<NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџб
8__inference_batch_normalization_57_layer_call_fn_81885959:;<NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџѕ
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188773?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ѕ
S__inference_batch_normalization_58_layer_call_and_return_conditional_losses_8188791?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
8__inference_batch_normalization_58_layer_call_fn_8188742?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЯ
8__inference_batch_normalization_58_layer_call_fn_8188755?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџП
F__inference_conv2d_37_layer_call_and_return_conditional_losses_8188307u+,8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ?
 
+__inference_conv2d_37_layer_call_fn_8188296j+,8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ "*'
unknownџџџџџџџџџ?О
F__inference_conv2d_38_layer_call_and_return_conditional_losses_8188399t128Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
+__inference_conv2d_38_layer_call_fn_8188388i128Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџэ
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_818856978IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
5__inference_conv2d_transpose_27_layer_call_fn_818851978IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџэ
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_8188729=>JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
5__inference_conv2d_transpose_28_layer_call_fn_8188679=>JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџц
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188504QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ц
F__inference_lambda_27_layer_call_and_return_conditional_losses_8188510QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Р
+__inference_lambda_27_layer_call_fn_8188493QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџР
+__inference_lambda_27_layer_call_fn_8188498QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџш
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188664RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ш
F__inference_lambda_28_layer_call_and_return_conditional_losses_8188670RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
+__inference_lambda_28_layer_call_fn_8188653RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
+__inference_lambda_28_layer_call_fn_8188658RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8188317ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_33_layer_call_fn_8188312RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_8188409ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_34_layer_call_fn_8188404RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџШ
%__inference_signature_wrapper_8187465+,-./0123456789:;<=>?@ABFЂC
Ђ 
<Њ9
7
input_59+(
input_59џџџџџџџџџ@":Њ7
5
Decoder*'
decoderџџџџџџџџџ@ї
M__inference_up_sampling2d_27_layer_call_and_return_conditional_losses_8188488ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_up_sampling2d_27_layer_call_fn_8188476RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_up_sampling2d_28_layer_call_and_return_conditional_losses_8188648ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_up_sampling2d_28_layer_call_fn_8188636RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ