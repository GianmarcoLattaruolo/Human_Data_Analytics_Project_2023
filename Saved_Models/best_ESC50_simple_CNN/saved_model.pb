г

нЌ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
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
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
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

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Є
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
~
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
:2*
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
:2*
dtype0

Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Рl2*&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes
:	Рl2*
dtype0

Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Рl2*&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes
:	Рl2*
dtype0

Adam/v/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_26/bias
{
)Adam/v/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/bias*
_output_shapes
:@*
dtype0

Adam/m/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_26/bias
{
)Adam/m/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/bias*
_output_shapes
:@*
dtype0

Adam/v/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/v/conv2d_26/kernel

+Adam/v/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/kernel*&
_output_shapes
: @*
dtype0

Adam/m/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/m/conv2d_26/kernel

+Adam/m/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/kernel*&
_output_shapes
: @*
dtype0

Adam/v/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_25/bias
{
)Adam/v/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_25/bias
{
)Adam/m/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_25/kernel

+Adam/v/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_25/kernel

+Adam/m/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/kernel*&
_output_shapes
: *
dtype0

Adam/v/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_24/bias
{
)Adam/v/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_24/bias
{
)Adam/m/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/conv2d_24/kernel

+Adam/v/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/conv2d_24/kernel

+Adam/m/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/kernel*&
_output_shapes
:*
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
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:2*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Рl2*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	Рl2*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0

conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
: *
dtype0

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
: *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:*
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:*
dtype0

serving_default_conv2d_24_inputPlaceholder*0
_output_shapes
:џџџџџџџџџxѕ*
dtype0*%
shape:џџџџџџџџџxѕ
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_24_inputconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_8/kerneldense_8/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1516607

NoOpNoOp
H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*НG
valueГGBАG BЉG

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
Ш
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
Ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op*

9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 

?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
І
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias*
<
0
1
'2
(3
64
75
K6
L7*
<
0
1
'2
(3
64
75
K6
L7*
* 
А
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
6
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_3* 
* 

Z
_variables
[_iterations
\_learning_rate
]_index_dict
^
_momentums
__velocities
`_update_step_xla*

aserving_default* 

0
1*

0
1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
`Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

ntrace_0* 

otrace_0* 

'0
(1*

'0
(1*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
`Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

|trace_0* 

}trace_0* 

60
71*

60
71*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

K0
L1*

K0
L1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

0
1*
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

[0
1
2
3
4
 5
Ё6
Ђ7
Ѓ8
Є9
Ѕ10
І11
Ї12
Ј13
Љ14
Њ15
Ћ16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
0
1
 2
Ђ3
Є4
І5
Ј6
Њ7*
D
0
1
Ё2
Ѓ3
Ѕ4
Ї5
Љ6
Ћ7*
r
Ќtrace_0
­trace_1
Ўtrace_2
Џtrace_3
Аtrace_4
Бtrace_5
Вtrace_6
Гtrace_7* 
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
<
Д	variables
Е	keras_api

Жtotal

Зcount*
M
И	variables
Й	keras_api

Кtotal

Лcount
М
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/conv2d_24/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_24/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_24/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_24/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_25/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_25/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_25/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_25/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_26/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_26/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_26/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_26/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

Ж0
З1*

Д	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

К0
Л1*

И	variables*
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
ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_8/kerneldense_8/bias	iterationlearning_rateAdam/m/conv2d_24/kernelAdam/v/conv2d_24/kernelAdam/m/conv2d_24/biasAdam/v/conv2d_24/biasAdam/m/conv2d_25/kernelAdam/v/conv2d_25/kernelAdam/m/conv2d_25/biasAdam/v/conv2d_25/biasAdam/m/conv2d_26/kernelAdam/v/conv2d_26/kernelAdam/m/conv2d_26/biasAdam/v/conv2d_26/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biastotal_1count_1totalcountConst*+
Tin$
"2 *
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
 __inference__traced_save_1517087
ю
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_8/kerneldense_8/bias	iterationlearning_rateAdam/m/conv2d_24/kernelAdam/v/conv2d_24/kernelAdam/m/conv2d_24/biasAdam/v/conv2d_24/biasAdam/m/conv2d_25/kernelAdam/v/conv2d_25/kernelAdam/m/conv2d_25/biasAdam/v/conv2d_25/biasAdam/m/conv2d_26/kernelAdam/v/conv2d_26/kernelAdam/m/conv2d_26/biasAdam/v/conv2d_26/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biastotal_1count_1totalcount**
Tin#
!2*
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
#__inference__traced_restore_1517187еі
­
L
$__inference__update_step_xla_1516743
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

i
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516853

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


л
*__inference_CNN_MFCC_layer_call_fn_1516445
conv2d_24_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	Рl2
	unknown_6:2
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input
П
N
2__inference_max_pooling2d_26_layer_call_fn_1516848

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
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277
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
Ш

)__inference_dense_8_layer_call_fn_1516873

inputs
unknown:	Рl2
	unknown_0:2
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџРl: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџРl
 
_user_specified_nameinputs
б
X
$__inference__update_step_xla_1516748
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:P L
&
_output_shapes
: @
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
б
X
$__inference__update_step_xla_1516728
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

i
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516793

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

џ
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516813

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ} i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ}
 
_user_specified_nameinputs

џ
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ>@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ>@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ> 
 
_user_specified_nameinputs

џ
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ} i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ}
 
_user_specified_nameinputs
-
н
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516723

inputsB
(conv2d_24_conv2d_readvariableop_resource:7
)conv2d_24_biasadd_readvariableop_resource:B
(conv2d_25_conv2d_readvariableop_resource: 7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: @7
)conv2d_26_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	Рl25
'dense_8_biasadd_readvariableop_resource:2
identityЂ conv2d_24/BiasAdd/ReadVariableOpЂconv2d_24/Conv2D/ReadVariableOpЂ conv2d_25/BiasAdd/ReadVariableOpЂconv2d_25/Conv2D/ReadVariableOpЂ conv2d_26/BiasAdd/ReadVariableOpЂconv2d_26/Conv2D/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOp
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћm
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћЎ
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ}*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} *
paddingSAME*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ} Ў
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ> *
ksize
*
paddingVALID*
strides

conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ш
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@*
paddingSAME*
strides

 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@l
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ>@Ў
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@6  
flatten_8/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	Рl2*
dtype0
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2ж
NoOpNoOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_1516733
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

i
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516823

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
Ь
Q
$__inference__update_step_xla_1516758
gradient
variable:	Рl2*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ2: *
	_noinline(:Q M
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

i
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265

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
Ж
G
+__inference_flatten_8_layer_call_fn_1516858

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџРl* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516864

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@6  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџРlY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѕ
 
+__inference_conv2d_24_layer_call_fn_1516772

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ<ћ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ<ћ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџxѕ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
П
N
2__inference_max_pooling2d_24_layer_call_fn_1516788

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
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253
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
Ъ#
Ј
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516367
conv2d_24_input+
conv2d_24_1516299:
conv2d_24_1516301:+
conv2d_25_1516317: 
conv2d_25_1516319: +
conv2d_26_1516335: @
conv2d_26_1516337:@"
dense_8_1516361:	Рl2
dense_8_1516363:2
identityЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_8/StatefulPartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputconv2d_24_1516299conv2d_24_1516301*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ<ћ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298ј
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_1516317conv2d_25_1516319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316ј
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265Ѕ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_1516335conv2d_26_1516337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334ј
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277т
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџРl* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_1516361dense_8_1516363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2д
NoOpNoOp"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input
Ъ#
Ј
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516395
conv2d_24_input+
conv2d_24_1516370:
conv2d_24_1516372:+
conv2d_25_1516376: 
conv2d_25_1516378: +
conv2d_26_1516382: @
conv2d_26_1516384:@"
dense_8_1516389:	Рl2
dense_8_1516391:2
identityЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_8/StatefulPartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputconv2d_24_1516370conv2d_24_1516372*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ<ћ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298ј
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_1516376conv2d_25_1516378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316ј
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265Ѕ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_1516382conv2d_26_1516384*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334ј
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277т
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџРl* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_1516389dense_8_1516391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2д
NoOpNoOp"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input
Џ#

E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516475

inputs+
conv2d_24_1516450:
conv2d_24_1516452:+
conv2d_25_1516456: 
conv2d_25_1516458: +
conv2d_26_1516462: @
conv2d_26_1516464:@"
dense_8_1516469:	Рl2
dense_8_1516471:2
identityЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_8/StatefulPartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_1516450conv2d_24_1516452*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ<ћ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298ј
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_1516456conv2d_25_1516458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316ј
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265Ѕ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_1516462conv2d_26_1516464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334ј
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277т
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџРl* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_1516469dense_8_1516471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2д
NoOpNoOp"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs

џ
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516843

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ>@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ>@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ> 
 
_user_specified_nameinputs
с	
ж
%__inference_signature_wrapper_1516607
conv2d_24_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	Рl2
	unknown_6:2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1516247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input
ю	
в
*__inference_CNN_MFCC_layer_call_fn_1516649

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	Рl2
	unknown_6:2
identityЂStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277

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
$__inference__update_step_xla_1516763
gradient
variable:2*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:2: *
	_noinline(:D @

_output_shapes
:2
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Џ#

E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516426

inputs+
conv2d_24_1516401:
conv2d_24_1516403:+
conv2d_25_1516407: 
conv2d_25_1516409: +
conv2d_26_1516413: @
conv2d_26_1516415:@"
dense_8_1516420:	Рl2
dense_8_1516422:2
identityЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_8/StatefulPartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_1516401conv2d_24_1516403*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ<ћ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298ј
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_1516407conv2d_25_1516409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316ј
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265Ѕ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_1516413conv2d_26_1516415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334ј
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516277т
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџРl* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_1516420dense_8_1516422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2д
NoOpNoOp"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
н
Щ
 __inference__traced_save_1517087
file_prefixA
'read_disablecopyonread_conv2d_24_kernel:5
'read_1_disablecopyonread_conv2d_24_bias:C
)read_2_disablecopyonread_conv2d_25_kernel: 5
'read_3_disablecopyonread_conv2d_25_bias: C
)read_4_disablecopyonread_conv2d_26_kernel: @5
'read_5_disablecopyonread_conv2d_26_bias:@:
'read_6_disablecopyonread_dense_8_kernel:	Рl23
%read_7_disablecopyonread_dense_8_bias:2,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: K
1read_10_disablecopyonread_adam_m_conv2d_24_kernel:K
1read_11_disablecopyonread_adam_v_conv2d_24_kernel:=
/read_12_disablecopyonread_adam_m_conv2d_24_bias:=
/read_13_disablecopyonread_adam_v_conv2d_24_bias:K
1read_14_disablecopyonread_adam_m_conv2d_25_kernel: K
1read_15_disablecopyonread_adam_v_conv2d_25_kernel: =
/read_16_disablecopyonread_adam_m_conv2d_25_bias: =
/read_17_disablecopyonread_adam_v_conv2d_25_bias: K
1read_18_disablecopyonread_adam_m_conv2d_26_kernel: @K
1read_19_disablecopyonread_adam_v_conv2d_26_kernel: @=
/read_20_disablecopyonread_adam_m_conv2d_26_bias:@=
/read_21_disablecopyonread_adam_v_conv2d_26_bias:@B
/read_22_disablecopyonread_adam_m_dense_8_kernel:	Рl2B
/read_23_disablecopyonread_adam_v_dense_8_kernel:	Рl2;
-read_24_disablecopyonread_adam_m_dense_8_bias:2;
-read_25_disablecopyonread_adam_v_dense_8_bias:2+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_24_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv2d_24_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv2d_24_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 Б
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_conv2d_25_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: {
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_conv2d_25_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_conv2d_25_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 Б
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_conv2d_26_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_conv2d_26_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_conv2d_26_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_8_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Рl2*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 Ё
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_8_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:2v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnRead1read_10_disablecopyonread_adam_m_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 Л
Read_10/ReadVariableOpReadVariableOp1read_10_disablecopyonread_adam_m_conv2d_24_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_adam_v_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 Л
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_adam_v_conv2d_24_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_adam_m_conv2d_24_bias"/device:CPU:0*
_output_shapes
 ­
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_adam_m_conv2d_24_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_v_conv2d_24_bias"/device:CPU:0*
_output_shapes
 ­
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_v_conv2d_24_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_adam_m_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 Л
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_adam_m_conv2d_25_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_adam_v_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 Л
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_adam_v_conv2d_25_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_m_conv2d_25_bias"/device:CPU:0*
_output_shapes
 ­
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_m_conv2d_25_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_v_conv2d_25_bias"/device:CPU:0*
_output_shapes
 ­
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_v_conv2d_25_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_adam_m_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 Л
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_adam_m_conv2d_26_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_v_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 Л
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_v_conv2d_26_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_m_conv2d_26_bias"/device:CPU:0*
_output_shapes
 ­
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_m_conv2d_26_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_v_conv2d_26_bias"/device:CPU:0*
_output_shapes
 ­
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_v_conv2d_26_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 В
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_dense_8_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Рl2*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 В
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_dense_8_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Рl2*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	Рl2
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_m_dense_8_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:2
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_adam_v_dense_8_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:2v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: И
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: љ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Ш
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516347

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@6  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџРlY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ю	
в
*__inference_CNN_MFCC_layer_call_fn_1516628

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	Рl2
	unknown_6:2
identityЂStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
ё
 
+__inference_conv2d_25_layer_call_fn_1516802

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516316w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ}: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ}
 
_user_specified_nameinputs
ё
 
+__inference_conv2d_26_layer_call_fn_1516832

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516334w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ>@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ> : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ> 
 
_user_specified_nameinputs

л
#__inference__traced_restore_1517187
file_prefix;
!assignvariableop_conv2d_24_kernel:/
!assignvariableop_1_conv2d_24_bias:=
#assignvariableop_2_conv2d_25_kernel: /
!assignvariableop_3_conv2d_25_bias: =
#assignvariableop_4_conv2d_26_kernel: @/
!assignvariableop_5_conv2d_26_bias:@4
!assignvariableop_6_dense_8_kernel:	Рl2-
assignvariableop_7_dense_8_bias:2&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: E
+assignvariableop_10_adam_m_conv2d_24_kernel:E
+assignvariableop_11_adam_v_conv2d_24_kernel:7
)assignvariableop_12_adam_m_conv2d_24_bias:7
)assignvariableop_13_adam_v_conv2d_24_bias:E
+assignvariableop_14_adam_m_conv2d_25_kernel: E
+assignvariableop_15_adam_v_conv2d_25_kernel: 7
)assignvariableop_16_adam_m_conv2d_25_bias: 7
)assignvariableop_17_adam_v_conv2d_25_bias: E
+assignvariableop_18_adam_m_conv2d_26_kernel: @E
+assignvariableop_19_adam_v_conv2d_26_kernel: @7
)assignvariableop_20_adam_m_conv2d_26_bias:@7
)assignvariableop_21_adam_v_conv2d_26_bias:@<
)assignvariableop_22_adam_m_dense_8_kernel:	Рl2<
)assignvariableop_23_adam_v_dense_8_kernel:	Рl25
'assignvariableop_24_adam_m_dense_8_bias:25
'assignvariableop_25_adam_v_dense_8_bias:2%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B К
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_24_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_24_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_25_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_25_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_26_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_26_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_8_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_8_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_m_conv2d_24_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adam_v_conv2d_24_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_conv2d_24_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_conv2d_24_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_m_conv2d_25_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_v_conv2d_25_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_conv2d_25_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_conv2d_25_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_m_conv2d_26_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_v_conv2d_26_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_m_conv2d_26_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_v_conv2d_26_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_8_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_8_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_8_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_8_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: а
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

џ
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516298

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ<ћw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџxѕ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
Є

і
D__inference_dense_8_layer_call_and_return_conditional_losses_1516884

inputs1
matmul_readvariableop_resource:	Рl2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Рl2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџРl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџРl
 
_user_specified_nameinputs


л
*__inference_CNN_MFCC_layer_call_fn_1516494
conv2d_24_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	Рl2
	unknown_6:2
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input

џ
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516783

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ<ћw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџxѕ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
-
н
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516686

inputsB
(conv2d_24_conv2d_readvariableop_resource:7
)conv2d_24_biasadd_readvariableop_resource:B
(conv2d_25_conv2d_readvariableop_resource: 7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: @7
)conv2d_26_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	Рl25
'dense_8_biasadd_readvariableop_resource:2
identityЂ conv2d_24/BiasAdd/ReadVariableOpЂconv2d_24/Conv2D/ReadVariableOpЂ conv2d_25/BiasAdd/ReadVariableOpЂconv2d_25/Conv2D/ReadVariableOpЂ conv2d_26/BiasAdd/ReadVariableOpЂconv2d_26/Conv2D/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOp
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћm
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћЎ
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ}*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} *
paddingSAME*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ} Ў
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ> *
ksize
*
paddingVALID*
strides

conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ш
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@*
paddingSAME*
strides

 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@l
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ>@Ў
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@6  
flatten_8/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	Рl2*
dtype0
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2ж
NoOpNoOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџxѕ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_1516753
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Є

і
D__inference_dense_8_layer_call_and_return_conditional_losses_1516360

inputs1
matmul_readvariableop_resource:	Рl2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Рl2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџРl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџРl
 
_user_specified_nameinputs
ь3
г
"__inference__wrapped_model_1516247
conv2d_24_inputK
1cnn_mfcc_conv2d_24_conv2d_readvariableop_resource:@
2cnn_mfcc_conv2d_24_biasadd_readvariableop_resource:K
1cnn_mfcc_conv2d_25_conv2d_readvariableop_resource: @
2cnn_mfcc_conv2d_25_biasadd_readvariableop_resource: K
1cnn_mfcc_conv2d_26_conv2d_readvariableop_resource: @@
2cnn_mfcc_conv2d_26_biasadd_readvariableop_resource:@B
/cnn_mfcc_dense_8_matmul_readvariableop_resource:	Рl2>
0cnn_mfcc_dense_8_biasadd_readvariableop_resource:2
identityЂ)CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOpЂ(CNN_MFCC/conv2d_24/Conv2D/ReadVariableOpЂ)CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOpЂ(CNN_MFCC/conv2d_25/Conv2D/ReadVariableOpЂ)CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOpЂ(CNN_MFCC/conv2d_26/Conv2D/ReadVariableOpЂ'CNN_MFCC/dense_8/BiasAdd/ReadVariableOpЂ&CNN_MFCC/dense_8/MatMul/ReadVariableOpЂ
(CNN_MFCC/conv2d_24/Conv2D/ReadVariableOpReadVariableOp1cnn_mfcc_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
CNN_MFCC/conv2d_24/Conv2DConv2Dconv2d_24_input0CNN_MFCC/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ*
paddingSAME*
strides

)CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp2cnn_mfcc_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0З
CNN_MFCC/conv2d_24/BiasAddBiasAdd"CNN_MFCC/conv2d_24/Conv2D:output:01CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћ
CNN_MFCC/conv2d_24/ReluRelu#CNN_MFCC/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ<ћР
!CNN_MFCC/max_pooling2d_24/MaxPoolMaxPool%CNN_MFCC/conv2d_24/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ}*
ksize
*
paddingVALID*
strides
Ђ
(CNN_MFCC/conv2d_25/Conv2D/ReadVariableOpReadVariableOp1cnn_mfcc_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0у
CNN_MFCC/conv2d_25/Conv2DConv2D*CNN_MFCC/max_pooling2d_24/MaxPool:output:00CNN_MFCC/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} *
paddingSAME*
strides

)CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp2cnn_mfcc_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж
CNN_MFCC/conv2d_25/BiasAddBiasAdd"CNN_MFCC/conv2d_25/Conv2D:output:01CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ} ~
CNN_MFCC/conv2d_25/ReluRelu#CNN_MFCC/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ} Р
!CNN_MFCC/max_pooling2d_25/MaxPoolMaxPool%CNN_MFCC/conv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ> *
ksize
*
paddingVALID*
strides
Ђ
(CNN_MFCC/conv2d_26/Conv2D/ReadVariableOpReadVariableOp1cnn_mfcc_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0у
CNN_MFCC/conv2d_26/Conv2DConv2D*CNN_MFCC/max_pooling2d_25/MaxPool:output:00CNN_MFCC/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@*
paddingSAME*
strides

)CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp2cnn_mfcc_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
CNN_MFCC/conv2d_26/BiasAddBiasAdd"CNN_MFCC/conv2d_26/Conv2D:output:01CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ>@~
CNN_MFCC/conv2d_26/ReluRelu#CNN_MFCC/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ>@Р
!CNN_MFCC/max_pooling2d_26/MaxPoolMaxPool%CNN_MFCC/conv2d_26/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
i
CNN_MFCC/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@6  Ї
CNN_MFCC/flatten_8/ReshapeReshape*CNN_MFCC/max_pooling2d_26/MaxPool:output:0!CNN_MFCC/flatten_8/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџРl
&CNN_MFCC/dense_8/MatMul/ReadVariableOpReadVariableOp/cnn_mfcc_dense_8_matmul_readvariableop_resource*
_output_shapes
:	Рl2*
dtype0Ј
CNN_MFCC/dense_8/MatMulMatMul#CNN_MFCC/flatten_8/Reshape:output:0.CNN_MFCC/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
'CNN_MFCC/dense_8/BiasAdd/ReadVariableOpReadVariableOp0cnn_mfcc_dense_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Љ
CNN_MFCC/dense_8/BiasAddBiasAdd!CNN_MFCC/dense_8/MatMul:product:0/CNN_MFCC/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2x
CNN_MFCC/dense_8/SoftmaxSoftmax!CNN_MFCC/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2q
IdentityIdentity"CNN_MFCC/dense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
NoOpNoOp*^CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOp)^CNN_MFCC/conv2d_24/Conv2D/ReadVariableOp*^CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOp)^CNN_MFCC/conv2d_25/Conv2D/ReadVariableOp*^CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOp)^CNN_MFCC/conv2d_26/Conv2D/ReadVariableOp(^CNN_MFCC/dense_8/BiasAdd/ReadVariableOp'^CNN_MFCC/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџxѕ: : : : : : : : 2V
)CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOp)CNN_MFCC/conv2d_24/BiasAdd/ReadVariableOp2T
(CNN_MFCC/conv2d_24/Conv2D/ReadVariableOp(CNN_MFCC/conv2d_24/Conv2D/ReadVariableOp2V
)CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOp)CNN_MFCC/conv2d_25/BiasAdd/ReadVariableOp2T
(CNN_MFCC/conv2d_25/Conv2D/ReadVariableOp(CNN_MFCC/conv2d_25/Conv2D/ReadVariableOp2V
)CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOp)CNN_MFCC/conv2d_26/BiasAdd/ReadVariableOp2T
(CNN_MFCC/conv2d_26/Conv2D/ReadVariableOp(CNN_MFCC/conv2d_26/Conv2D/ReadVariableOp2R
'CNN_MFCC/dense_8/BiasAdd/ReadVariableOp'CNN_MFCC/dense_8/BiasAdd/ReadVariableOp2P
&CNN_MFCC/dense_8/MatMul/ReadVariableOp&CNN_MFCC/dense_8/MatMul/ReadVariableOp:a ]
0
_output_shapes
:џџџџџџџџџxѕ
)
_user_specified_nameconv2d_24_input

i
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516253

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
П
N
2__inference_max_pooling2d_25_layer_call_fn_1516818

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
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516265
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
б
X
$__inference__update_step_xla_1516738
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*У
serving_defaultЏ
T
conv2d_24_inputA
!serving_default_conv2d_24_input:0џџџџџџџџџxѕ;
dense_80
StatefulPartitionedCall:0џџџџџџџџџ2tensorflow/serving/predict:вы
Ж
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
н
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
н
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias"
_tf_keras_layer
X
0
1
'2
(3
64
75
K6
L7"
trackable_list_wrapper
X
0
1
'2
(3
64
75
K6
L7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
г
Rtrace_0
Strace_1
Ttrace_2
Utrace_32ш
*__inference_CNN_MFCC_layer_call_fn_1516445
*__inference_CNN_MFCC_layer_call_fn_1516494
*__inference_CNN_MFCC_layer_call_fn_1516628
*__inference_CNN_MFCC_layer_call_fn_1516649Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
П
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_32д
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516367
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516395
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516686
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516723Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zVtrace_0zWtrace_1zXtrace_2zYtrace_3
еBв
"__inference__wrapped_model_1516247conv2d_24_input"
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
Z
_variables
[_iterations
\_learning_rate
]_index_dict
^
_momentums
__velocities
`_update_step_xla"
experimentalOptimizer
,
aserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
х
gtrace_02Ш
+__inference_conv2d_24_layer_call_fn_1516772
В
FullArgSpec
args

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
annotationsЊ *
 zgtrace_0

htrace_02у
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516783
В
FullArgSpec
args

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
annotationsЊ *
 zhtrace_0
*:(2conv2d_24/kernel
:2conv2d_24/bias
Њ2ЇЄ
В
FullArgSpec
args
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
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ь
ntrace_02Я
2__inference_max_pooling2d_24_layer_call_fn_1516788
В
FullArgSpec
args

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
annotationsЊ *
 zntrace_0

otrace_02ъ
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516793
В
FullArgSpec
args

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
annotationsЊ *
 zotrace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
х
utrace_02Ш
+__inference_conv2d_25_layer_call_fn_1516802
В
FullArgSpec
args

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
annotationsЊ *
 zutrace_0

vtrace_02у
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516813
В
FullArgSpec
args

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
annotationsЊ *
 zvtrace_0
*:( 2conv2d_25/kernel
: 2conv2d_25/bias
Њ2ЇЄ
В
FullArgSpec
args
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
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ь
|trace_02Я
2__inference_max_pooling2d_25_layer_call_fn_1516818
В
FullArgSpec
args

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
annotationsЊ *
 z|trace_0

}trace_02ъ
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516823
В
FullArgSpec
args

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
annotationsЊ *
 z}trace_0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
А
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ч
trace_02Ш
+__inference_conv2d_26_layer_call_fn_1516832
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02у
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516843
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
*:( @2conv2d_26/kernel
:@2conv2d_26/bias
Њ2ЇЄ
В
FullArgSpec
args
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
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
2__inference_max_pooling2d_26_layer_call_fn_1516848
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ъ
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516853
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ч
trace_02Ш
+__inference_flatten_8_layer_call_fn_1516858
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02у
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516864
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_dense_8_layer_call_fn_1516873
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02с
D__inference_dense_8_layer_call_and_return_conditional_losses_1516884
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
!:	Рl22dense_8/kernel
:22dense_8/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
*__inference_CNN_MFCC_layer_call_fn_1516445conv2d_24_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
*__inference_CNN_MFCC_layer_call_fn_1516494conv2d_24_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
*__inference_CNN_MFCC_layer_call_fn_1516628inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
*__inference_CNN_MFCC_layer_call_fn_1516649inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516367conv2d_24_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516395conv2d_24_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516686inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516723inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ў
[0
1
2
3
4
 5
Ё6
Ђ7
Ѓ8
Є9
Ѕ10
І11
Ї12
Ј13
Љ14
Њ15
Ћ16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
0
1
 2
Ђ3
Є4
І5
Ј6
Њ7"
trackable_list_wrapper
`
0
1
Ё2
Ѓ3
Ѕ4
Ї5
Љ6
Ћ7"
trackable_list_wrapper
Х
Ќtrace_0
­trace_1
Ўtrace_2
Џtrace_3
Аtrace_4
Бtrace_5
Вtrace_6
Гtrace_72т
$__inference__update_step_xla_1516728
$__inference__update_step_xla_1516733
$__inference__update_step_xla_1516738
$__inference__update_step_xla_1516743
$__inference__update_step_xla_1516748
$__inference__update_step_xla_1516753
$__inference__update_step_xla_1516758
$__inference__update_step_xla_1516763Џ
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 0zЌtrace_0z­trace_1zЎtrace_2zЏtrace_3zАtrace_4zБtrace_5zВtrace_6zГtrace_7
дBб
%__inference_signature_wrapper_1516607conv2d_24_input"
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
еBв
+__inference_conv2d_24_layer_call_fn_1516772inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516783inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
мBй
2__inference_max_pooling2d_24_layer_call_fn_1516788inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
їBє
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516793inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
еBв
+__inference_conv2d_25_layer_call_fn_1516802inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516813inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
мBй
2__inference_max_pooling2d_25_layer_call_fn_1516818inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
їBє
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516823inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
еBв
+__inference_conv2d_26_layer_call_fn_1516832inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516843inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
мBй
2__inference_max_pooling2d_26_layer_call_fn_1516848inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
їBє
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516853inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
еBв
+__inference_flatten_8_layer_call_fn_1516858inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516864inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_dense_8_layer_call_fn_1516873inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_8_layer_call_and_return_conditional_losses_1516884inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
R
Д	variables
Е	keras_api

Жtotal

Зcount"
_tf_keras_metric
c
И	variables
Й	keras_api

Кtotal

Лcount
М
_fn_kwargs"
_tf_keras_metric
/:-2Adam/m/conv2d_24/kernel
/:-2Adam/v/conv2d_24/kernel
!:2Adam/m/conv2d_24/bias
!:2Adam/v/conv2d_24/bias
/:- 2Adam/m/conv2d_25/kernel
/:- 2Adam/v/conv2d_25/kernel
!: 2Adam/m/conv2d_25/bias
!: 2Adam/v/conv2d_25/bias
/:- @2Adam/m/conv2d_26/kernel
/:- @2Adam/v/conv2d_26/kernel
!:@2Adam/m/conv2d_26/bias
!:@2Adam/v/conv2d_26/bias
&:$	Рl22Adam/m/dense_8/kernel
&:$	Рl22Adam/v/dense_8/kernel
:22Adam/m/dense_8/bias
:22Adam/v/dense_8/bias
яBь
$__inference__update_step_xla_1516728gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516733gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516738gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516743gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516748gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516753gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516758gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
яBь
$__inference__update_step_xla_1516763gradientvariable"­
ІВЂ
FullArgSpec*
args"

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
annotationsЊ *
 
0
Ж0
З1"
trackable_list_wrapper
.
Д	variables"
_generic_user_object
:  (2total
:  (2count
0
К0
Л1"
trackable_list_wrapper
.
И	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЭ
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516367'(67KLIЂF
?Ђ<
2/
conv2d_24_inputџџџџџџџџџxѕ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ2
 Э
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516395'(67KLIЂF
?Ђ<
2/
conv2d_24_inputџџџџџџџџџxѕ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ2
 У
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516686z'(67KL@Ђ=
6Ђ3
)&
inputsџџџџџџџџџxѕ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ2
 У
E__inference_CNN_MFCC_layer_call_and_return_conditional_losses_1516723z'(67KL@Ђ=
6Ђ3
)&
inputsџџџџџџџџџxѕ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ2
 І
*__inference_CNN_MFCC_layer_call_fn_1516445x'(67KLIЂF
?Ђ<
2/
conv2d_24_inputџџџџџџџџџxѕ
p

 
Њ "!
unknownџџџџџџџџџ2І
*__inference_CNN_MFCC_layer_call_fn_1516494x'(67KLIЂF
?Ђ<
2/
conv2d_24_inputџџџџџџџџџxѕ
p 

 
Њ "!
unknownџџџџџџџџџ2
*__inference_CNN_MFCC_layer_call_fn_1516628o'(67KL@Ђ=
6Ђ3
)&
inputsџџџџџџџџџxѕ
p

 
Њ "!
unknownџџџџџџџџџ2
*__inference_CNN_MFCC_layer_call_fn_1516649o'(67KL@Ђ=
6Ђ3
)&
inputsџџџџџџџџџxѕ
p 

 
Њ "!
unknownџџџџџџџџџ2І
$__inference__update_step_xla_1516728~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`рѕЭЇњ=
Њ "
 
$__inference__update_step_xla_1516733f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЭгЇњ=
Њ "
 І
$__inference__update_step_xla_1516738~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
` аЭЇњ=
Њ "
 
$__inference__update_step_xla_1516743f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
` ЃЭЇњ=
Њ "
 І
$__inference__update_step_xla_1516748~xЂu
nЂk
!
gradient @
<9	%Ђ"
њ @

p
` VariableSpec 
`рѓЬЇњ=
Њ "
 
$__inference__update_step_xla_1516753f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`р­ЬЇњ=
Њ "
  
$__inference__update_step_xla_1516758xrЂo
hЂe
"
gradientџџџџџџџџџ2
52	Ђ
њ	Рl2

p
` VariableSpec 
` ­гЇњ=
Њ "
 
$__inference__update_step_xla_1516763f`Ђ]
VЂS

gradient2
0-	Ђ
њ2

p
` VariableSpec 
` ћЇњ=
Њ "
 Ї
"__inference__wrapped_model_1516247'(67KLAЂ>
7Ђ4
2/
conv2d_24_inputџџџџџџџџџxѕ
Њ "1Њ.
,
dense_8!
dense_8џџџџџџџџџ2П
F__inference_conv2d_24_layer_call_and_return_conditional_losses_1516783u8Ђ5
.Ђ+
)&
inputsџџџџџџџџџxѕ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ<ћ
 
+__inference_conv2d_24_layer_call_fn_1516772j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџxѕ
Њ "*'
unknownџџџџџџџџџ<ћН
F__inference_conv2d_25_layer_call_and_return_conditional_losses_1516813s'(7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ}
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ} 
 
+__inference_conv2d_25_layer_call_fn_1516802h'(7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ}
Њ ")&
unknownџџџџџџџџџ} Н
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1516843s677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ> 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ>@
 
+__inference_conv2d_26_layer_call_fn_1516832h677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ> 
Њ ")&
unknownџџџџџџџџџ>@Ќ
D__inference_dense_8_layer_call_and_return_conditional_losses_1516884dKL0Ђ-
&Ђ#
!
inputsџџџџџџџџџРl
Њ ",Ђ)
"
tensor_0џџџџџџџџџ2
 
)__inference_dense_8_layer_call_fn_1516873YKL0Ђ-
&Ђ#
!
inputsџџџџџџџџџРl
Њ "!
unknownџџџџџџџџџ2В
F__inference_flatten_8_layer_call_and_return_conditional_losses_1516864h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
tensor_0џџџџџџџџџРl
 
+__inference_flatten_8_layer_call_fn_1516858]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ ""
unknownџџџџџџџџџРlї
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_1516793ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_24_layer_call_fn_1516788RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_1516823ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_25_layer_call_fn_1516818RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_1516853ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_26_layer_call_fn_1516848RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџН
%__inference_signature_wrapper_1516607'(67KLTЂQ
Ђ 
JЊG
E
conv2d_24_input2/
conv2d_24_inputџџџџџџџџџxѕ"1Њ.
,
dense_8!
dense_8џџџџџџџџџ2