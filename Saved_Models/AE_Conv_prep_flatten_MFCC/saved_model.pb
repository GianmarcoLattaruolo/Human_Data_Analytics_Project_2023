��
��
.
Abs
x"T
y"T"
Ttype:

2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
�
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
$
DisableCopyOnRead
resource�
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
�
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
Adam/conv2d_transpose_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_8/bias/v
�
2Adam/conv2d_transpose_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/v*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_8/kernel/v
�
4Adam/conv2d_transpose_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_7/bias/v
�
2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/v*
_output_shapes
: *
dtype0
�
 Adam/conv2d_transpose_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*1
shared_name" Adam/conv2d_transpose_7/kernel/v
�
4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/v*&
_output_shapes
: 0*
dtype0
�
Adam/conv2d_transpose_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name Adam/conv2d_transpose_6/bias/v
�
2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/v*
_output_shapes
:0*
dtype0
�
 Adam/conv2d_transpose_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" Adam/conv2d_transpose_6/kernel/v
�
4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/v*&
_output_shapes
:00*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:0*
dtype0
�
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

: 0*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0 *&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:0 *
dtype0
�
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:0*
dtype0
�
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*'
shared_nameAdam/conv2d_8/kernel/v
�
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
: 0*
dtype0
�
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_7/kernel/v
�
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v
�
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_transpose_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_8/bias/m
�
2Adam/conv2d_transpose_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/m*
_output_shapes
:*
dtype0
�
 Adam/conv2d_transpose_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_8/kernel/m
�
4Adam/conv2d_transpose_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_transpose_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_7/bias/m
�
2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/m*
_output_shapes
: *
dtype0
�
 Adam/conv2d_transpose_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*1
shared_name" Adam/conv2d_transpose_7/kernel/m
�
4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/m*&
_output_shapes
: 0*
dtype0
�
Adam/conv2d_transpose_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name Adam/conv2d_transpose_6/bias/m
�
2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/m*
_output_shapes
:0*
dtype0
�
 Adam/conv2d_transpose_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*1
shared_name" Adam/conv2d_transpose_6/kernel/m
�
4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/m*&
_output_shapes
:00*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:0*
dtype0
�
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

: 0*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0 *&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:0 *
dtype0
�
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:0*
dtype0
�
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*'
shared_nameAdam/conv2d_8/kernel/m
�
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
: 0*
dtype0
�
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_7/kernel/m
�
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m
�
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_8/bias

+conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_8/kernel
�
-conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0**
shared_nameconv2d_transpose_7/kernel
�
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
: 0*
dtype0
�
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
:0*
dtype0
�
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00**
shared_nameconv2d_transpose_6/kernel
�
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
:00*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:0*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: 0*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0 *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:0 *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:0*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
: *
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_9Placeholder*0
_output_shapes
:���������@�*
dtype0*%
shape:���������@�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_3686059

NoOpNoOp
ǜ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B� B�
�
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
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories*
�
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"layer_with_weights-2
"layer-4
#layer-5
$layer_with_weights-3
$layer-6
%layer-7
&layer-8
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
#-_self_saveable_object_factories*
z
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15*
z
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 
�
Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_rate.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�<m�=m�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�<v�=v�*

Pserving_default* 
* 
* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

.kernel
/bias
#W_self_saveable_object_factories
 X_jit_compiled_convolution_op*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories* 
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

0kernel
1bias
#f_self_saveable_object_factories
 g_jit_compiled_convolution_op*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
#n_self_saveable_object_factories* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

2kernel
3bias
#u_self_saveable_object_factories
 v_jit_compiled_convolution_op*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
#}_self_saveable_object_factories* 
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

4kernel
5bias
$�_self_saveable_object_factories*
<
.0
/1
02
13
24
35
46
57*
<
.0
/1
02
13
24
35
46
57*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

6kernel
7bias
$�_self_saveable_object_factories*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

8kernel
9bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

:kernel
;bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

<kernel
=bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
<
60
71
82
93
:4
;5
<6
=7*
<
60
71
82
93
:4
;5
<6
=7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
OI
VARIABLE_VALUEconv2d_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_8/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_6/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_6/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_7/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_7/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_8/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_8/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

�0
�1*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

.0
/1*

.0
/1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

00
11*

00
11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
�activity_regularizer_fn
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
<
0
1
2
3
4
5
6
7*
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
60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

80
91*

80
91*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

:0
;1*

:0
;1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
C
0
1
 2
!3
"4
#5
$6
%7
&8*
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
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
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

�trace_0* 

�trace_0* 
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
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_6/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_7/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_7/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_8/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_8/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_4/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_4/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_5/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_6/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_7/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_7/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_8/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_8/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_4/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_4/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_5/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/m Adam/conv2d_transpose_8/kernel/mAdam/conv2d_transpose_8/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/v Adam/conv2d_transpose_8/kernel/vAdam/conv2d_transpose_8/bias/vConst*F
Tin?
=2;*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_3687520
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/m Adam/conv2d_transpose_8/kernel/mAdam/conv2d_transpose_8/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/v Adam/conv2d_transpose_8/kernel/vAdam/conv2d_transpose_8/bias/v*E
Tin>
<2:*
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_3687701��
��
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686276

inputsI
/encoder_conv2d_6_conv2d_readvariableop_resource:>
0encoder_conv2d_6_biasadd_readvariableop_resource:I
/encoder_conv2d_7_conv2d_readvariableop_resource: >
0encoder_conv2d_7_biasadd_readvariableop_resource: I
/encoder_conv2d_8_conv2d_readvariableop_resource: 0>
0encoder_conv2d_8_biasadd_readvariableop_resource:0@
.encoder_dense_4_matmul_readvariableop_resource:0 =
/encoder_dense_4_biasadd_readvariableop_resource: @
.decoder_dense_5_matmul_readvariableop_resource: 0=
/decoder_dense_5_biasadd_readvariableop_resource:0]
Cdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:00H
:decoder_conv2d_transpose_6_biasadd_readvariableop_resource:0]
Cdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0H
:decoder_conv2d_transpose_7_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_8_biasadd_readvariableop_resource:
identity

identity_1��1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp�1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp�1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp�&Decoder/dense_5/BiasAdd/ReadVariableOp�%Decoder/dense_5/MatMul/ReadVariableOp�'Encoder/conv2d_6/BiasAdd/ReadVariableOp�&Encoder/conv2d_6/Conv2D/ReadVariableOp�'Encoder/conv2d_7/BiasAdd/ReadVariableOp�&Encoder/conv2d_7/Conv2D/ReadVariableOp�'Encoder/conv2d_8/BiasAdd/ReadVariableOp�&Encoder/conv2d_8/Conv2D/ReadVariableOp�&Encoder/dense_4/BiasAdd/ReadVariableOp�%Encoder/dense_4/MatMul/ReadVariableOp�
&Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Encoder/conv2d_6/Conv2DConv2Dinputs.Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
'Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/conv2d_6/BiasAddBiasAdd Encoder/conv2d_6/Conv2D:output:0/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @z
Encoder/conv2d_6/TanhTanh!Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� @�
Encoder/max_pooling2d_6/MaxPoolMaxPoolEncoder/conv2d_6/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
&Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Encoder/conv2d_7/Conv2DConv2D(Encoder/max_pooling2d_6/MaxPool:output:0.Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Encoder/conv2d_7/BiasAddBiasAdd Encoder/conv2d_7/Conv2D:output:0/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
Encoder/conv2d_7/TanhTanh!Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
Encoder/max_pooling2d_7/MaxPoolMaxPoolEncoder/conv2d_7/Tanh:y:0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
&Encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Encoder/conv2d_8/Conv2DConv2D(Encoder/max_pooling2d_7/MaxPool:output:0.Encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
'Encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
Encoder/conv2d_8/BiasAddBiasAdd Encoder/conv2d_8/Conv2D:output:0/Encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0z
Encoder/conv2d_8/TanhTanh!Encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
Encoder/max_pooling2d_8/MaxPoolMaxPoolEncoder/conv2d_8/Tanh:y:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   �
Encoder/flatten_2/ReshapeReshape(Encoder/max_pooling2d_8/MaxPool:output:0 Encoder/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������0�
%Encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype0�
Encoder/dense_4/MatMulMatMul"Encoder/flatten_2/Reshape:output:0-Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Encoder/dense_4/BiasAddBiasAdd Encoder/dense_4/MatMul:product:0.Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
Encoder/dense_4/TanhTanh Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� z
'Encoder/dense_4/ActivityRegularizer/AbsAbsEncoder/dense_4/Tanh:y:0*
T0*'
_output_shapes
:��������� z
)Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'Encoder/dense_4/ActivityRegularizer/SumSum+Encoder/dense_4/ActivityRegularizer/Abs:y:02Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'Encoder/dense_4/ActivityRegularizer/mulMul2Encoder/dense_4/ActivityRegularizer/mul/x:output:00Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
)Encoder/dense_4/ActivityRegularizer/ShapeShapeEncoder/dense_4/Tanh:y:0*
T0*
_output_shapes
::���
7Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_4/ActivityRegularizer/Shape:output:0@Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(Encoder/dense_4/ActivityRegularizer/CastCast:Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+Encoder/dense_4/ActivityRegularizer/truedivRealDiv+Encoder/dense_4/ActivityRegularizer/mul:z:0,Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
%Decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype0�
Decoder/dense_5/MatMulMatMulEncoder/dense_4/Tanh:y:0-Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
&Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
Decoder/dense_5/BiasAddBiasAdd Decoder/dense_5/MatMul:product:0.Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0p
Decoder/dense_5/TanhTanh Decoder/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������0m
Decoder/reshape_2/ShapeShapeDecoder/dense_5/Tanh:y:0*
T0*
_output_shapes
::��o
%Decoder/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Decoder/reshape_2/strided_sliceStridedSlice Decoder/reshape_2/Shape:output:0.Decoder/reshape_2/strided_slice/stack:output:00Decoder/reshape_2/strided_slice/stack_1:output:00Decoder/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
Decoder/reshape_2/Reshape/shapePack(Decoder/reshape_2/strided_slice:output:0*Decoder/reshape_2/Reshape/shape/1:output:0*Decoder/reshape_2/Reshape/shape/2:output:0*Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Decoder/reshape_2/ReshapeReshapeDecoder/dense_5/Tanh:y:0(Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������0�
 Decoder/conv2d_transpose_6/ShapeShape"Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_6/strided_sliceStridedSlice)Decoder/conv2d_transpose_6/Shape:output:07Decoder/conv2d_transpose_6/strided_slice/stack:output:09Decoder/conv2d_transpose_6/strided_slice/stack_1:output:09Decoder/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0�
 Decoder/conv2d_transpose_6/stackPack1Decoder/conv2d_transpose_6/strided_slice:output:0+Decoder/conv2d_transpose_6/stack/1:output:0+Decoder/conv2d_transpose_6/stack/2:output:0+Decoder/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_6/strided_slice_1StridedSlice)Decoder/conv2d_transpose_6/stack:output:09Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
+Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_6/stack:output:0BDecoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
"Decoder/conv2d_transpose_6/BiasAddBiasAdd4Decoder/conv2d_transpose_6/conv2d_transpose:output:09Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0�
Decoder/conv2d_transpose_6/TanhTanh+Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������0n
Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_6/mulMul&Decoder/up_sampling2d_6/Const:output:0(Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_6/Tanh:y:0Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:���������0*
half_pixel_centers(�
 Decoder/conv2d_transpose_7/ShapeShapeEDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_7/strided_sliceStridedSlice)Decoder/conv2d_transpose_7/Shape:output:07Decoder/conv2d_transpose_7/strided_slice/stack:output:09Decoder/conv2d_transpose_7/strided_slice/stack_1:output:09Decoder/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
 Decoder/conv2d_transpose_7/stackPack1Decoder/conv2d_transpose_7/strided_slice:output:0+Decoder/conv2d_transpose_7/stack/1:output:0+Decoder/conv2d_transpose_7/stack/2:output:0+Decoder/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_7/strided_slice_1StridedSlice)Decoder/conv2d_transpose_7/stack:output:09Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
+Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_7/stack:output:0BDecoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"Decoder/conv2d_transpose_7/BiasAddBiasAdd4Decoder/conv2d_transpose_7/conv2d_transpose:output:09Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
Decoder/conv2d_transpose_7/TanhTanh+Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� n
Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_7/mulMul&Decoder/up_sampling2d_7/Const:output:0(Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_7/Tanh:y:0Decoder/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:���������$$ *
half_pixel_centers(�
 Decoder/conv2d_transpose_8/ShapeShapeEDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_8/strided_sliceStridedSlice)Decoder/conv2d_transpose_8/Shape:output:07Decoder/conv2d_transpose_8/strided_slice/stack:output:09Decoder/conv2d_transpose_8/strided_slice/stack_1:output:09Decoder/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Hd
"Decoder/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Hd
"Decoder/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 Decoder/conv2d_transpose_8/stackPack1Decoder/conv2d_transpose_8/strided_slice:output:0+Decoder/conv2d_transpose_8/stack/1:output:0+Decoder/conv2d_transpose_8/stack/2:output:0+Decoder/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_8/strided_slice_1StridedSlice)Decoder/conv2d_transpose_8/stack:output:09Decoder/conv2d_transpose_8/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_8/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
+Decoder/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_8/stack:output:0BDecoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:���������HH*
paddingSAME*
strides
�
1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"Decoder/conv2d_transpose_8/BiasAddBiasAdd4Decoder/conv2d_transpose_8/conv2d_transpose:output:09Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HH�
Decoder/conv2d_transpose_8/TanhTanh+Decoder/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������HHn
Decoder/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"H   H   p
Decoder/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_8/mulMul&Decoder/up_sampling2d_8/Const:output:0(Decoder/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_8/Tanh:y:0Decoder/up_sampling2d_8/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(o
Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
(Decoder/resizing_2/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(�
IdentityIdentity9Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:���������@�o

Identity_1Identity/Encoder/dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp2^Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp'^Decoder/dense_5/BiasAdd/ReadVariableOp&^Decoder/dense_5/MatMul/ReadVariableOp(^Encoder/conv2d_6/BiasAdd/ReadVariableOp'^Encoder/conv2d_6/Conv2D/ReadVariableOp(^Encoder/conv2d_7/BiasAdd/ReadVariableOp'^Encoder/conv2d_7/Conv2D/ReadVariableOp(^Encoder/conv2d_8/BiasAdd/ReadVariableOp'^Encoder/conv2d_8/Conv2D/ReadVariableOp'^Encoder/dense_4/BiasAdd/ReadVariableOp&^Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2f
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_5/BiasAdd/ReadVariableOp&Decoder/dense_5/BiasAdd/ReadVariableOp2N
%Decoder/dense_5/MatMul/ReadVariableOp%Decoder/dense_5/MatMul/ReadVariableOp2R
'Encoder/conv2d_6/BiasAdd/ReadVariableOp'Encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_6/Conv2D/ReadVariableOp&Encoder/conv2d_6/Conv2D/ReadVariableOp2R
'Encoder/conv2d_7/BiasAdd/ReadVariableOp'Encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_7/Conv2D/ReadVariableOp&Encoder/conv2d_7/Conv2D/ReadVariableOp2R
'Encoder/conv2d_8/BiasAdd/ReadVariableOp'Encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_8/Conv2D/ReadVariableOp&Encoder/conv2d_8/Conv2D/ReadVariableOp2P
&Encoder/dense_4/BiasAdd/ReadVariableOp&Encoder/dense_4/BiasAdd/ReadVariableOp2N
%Encoder/dense_4/MatMul/ReadVariableOp%Encoder/dense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685938
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
	unknown_7: 0
	unknown_8:0#
	unknown_9:00

unknown_10:0$

unknown_11: 0

unknown_12: $

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:���������@�: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685902x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�
�
*__inference_conv2d_8_layer_call_fn_3686862

inputs!
unknown: 0
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685824

inputs)
encoder_3685787:
encoder_3685789:)
encoder_3685791: 
encoder_3685793: )
encoder_3685795: 0
encoder_3685797:0!
encoder_3685799:0 
encoder_3685801: !
decoder_3685805: 0
decoder_3685807:0)
decoder_3685809:00
decoder_3685811:0)
decoder_3685813: 0
decoder_3685815: )
decoder_3685817: 
decoder_3685819:
identity

identity_1��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_3685787encoder_3685789encoder_3685791encoder_3685793encoder_3685795encoder_3685797encoder_3685799encoder_3685801*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685069�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_3685805decoder_3685807decoder_3685809decoder_3685811decoder_3685813decoder_3685815decoder_3685817decoder_3685819*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685567�
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�)
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685535
input_8!
dense_5_3685509: 0
dense_5_3685511:04
conv2d_transpose_6_3685515:00(
conv2d_transpose_6_3685517:04
conv2d_transpose_7_3685521: 0(
conv2d_transpose_7_3685523: 4
conv2d_transpose_8_3685527: (
conv2d_transpose_8_3685529:
identity��*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�*conv2d_transpose_8/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_5_3685509dense_5_3685511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457�
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_6_3685515conv2d_transpose_6_3685517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285�
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_3685521conv2d_transpose_7_3685523*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349�
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372�
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_8_3685527conv2d_transpose_8_3685529*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413�
up_sampling2d_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436�
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_8
�0
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3685069

inputs*
conv2d_6_3685035:
conv2d_6_3685037:*
conv2d_7_3685041: 
conv2d_7_3685043: *
conv2d_8_3685047: 0
conv2d_8_3685049:0!
dense_4_3685054:0 
dense_4_3685056: 
identity

identity_1�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_3685035conv2d_6_3685037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_3685041conv2d_7_3685043*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_3685047conv2d_8_3685049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950�
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_3685054dense_4_3685056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976�
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8� *9
f4R2
0__inference_dense_4_activity_regularizer_3684899�
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686097

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
	unknown_7: 0
	unknown_8:0#
	unknown_9:00

unknown_10:0$

unknown_11: 0

unknown_12: $

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:���������@�: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685824x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�)
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685506
input_8!
dense_5_3685458: 0
dense_5_3685460:04
conv2d_transpose_6_3685479:00(
conv2d_transpose_6_3685481:04
conv2d_transpose_7_3685485: 0(
conv2d_transpose_7_3685487: 4
conv2d_transpose_8_3685491: (
conv2d_transpose_8_3685493:
identity��*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�*conv2d_transpose_8/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_5_3685458dense_5_3685460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457�
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_6_3685479conv2d_transpose_6_3685481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285�
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_3685485conv2d_transpose_7_3685487*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349�
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372�
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_8_3685491conv2d_transpose_8_3685493*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413�
up_sampling2d_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436�
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_8
�
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3687084

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_3686894

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������0X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685781
input_9)
encoder_3685744:
encoder_3685746:)
encoder_3685748: 
encoder_3685750: )
encoder_3685752: 0
encoder_3685754:0!
encoder_3685756:0 
encoder_3685758: !
decoder_3685762: 0
decoder_3685764:0)
decoder_3685766:00
decoder_3685768:0)
decoder_3685770: 0
decoder_3685772: )
decoder_3685774: 
decoder_3685776:
identity

identity_1��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_3685744encoder_3685746encoder_3685748encoder_3685750encoder_3685752encoder_3685754encoder_3685756encoder_3685758*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685128�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_3685762decoder_3685764decoder_3685766decoder_3685768decoder_3685770decoder_3685772decoder_3685774decoder_3685776*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685617�
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�0
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3684992
input_7*
conv2d_6_3684915:
conv2d_6_3684917:*
conv2d_7_3684933: 
conv2d_7_3684935: *
conv2d_8_3684951: 0
conv2d_8_3684953:0!
dense_4_3684977:0 
dense_4_3684979: 
identity

identity_1�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_6_3684915conv2d_6_3684917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_3684933conv2d_7_3684935*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_3684951conv2d_8_3684953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950�
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_3684977dense_4_3684979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976�
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8� *9
f4R2
0__inference_dense_4_activity_regularizer_3684899�
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_7
��
�$
#__inference__traced_restore_3687701
file_prefix:
 assignvariableop_conv2d_6_kernel:.
 assignvariableop_1_conv2d_6_bias:<
"assignvariableop_2_conv2d_7_kernel: .
 assignvariableop_3_conv2d_7_bias: <
"assignvariableop_4_conv2d_8_kernel: 0.
 assignvariableop_5_conv2d_8_bias:03
!assignvariableop_6_dense_4_kernel:0 -
assignvariableop_7_dense_4_bias: 3
!assignvariableop_8_dense_5_kernel: 0-
assignvariableop_9_dense_5_bias:0G
-assignvariableop_10_conv2d_transpose_6_kernel:009
+assignvariableop_11_conv2d_transpose_6_bias:0G
-assignvariableop_12_conv2d_transpose_7_kernel: 09
+assignvariableop_13_conv2d_transpose_7_bias: G
-assignvariableop_14_conv2d_transpose_8_kernel: 9
+assignvariableop_15_conv2d_transpose_8_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: #
assignvariableop_23_total: #
assignvariableop_24_count: D
*assignvariableop_25_adam_conv2d_6_kernel_m:6
(assignvariableop_26_adam_conv2d_6_bias_m:D
*assignvariableop_27_adam_conv2d_7_kernel_m: 6
(assignvariableop_28_adam_conv2d_7_bias_m: D
*assignvariableop_29_adam_conv2d_8_kernel_m: 06
(assignvariableop_30_adam_conv2d_8_bias_m:0;
)assignvariableop_31_adam_dense_4_kernel_m:0 5
'assignvariableop_32_adam_dense_4_bias_m: ;
)assignvariableop_33_adam_dense_5_kernel_m: 05
'assignvariableop_34_adam_dense_5_bias_m:0N
4assignvariableop_35_adam_conv2d_transpose_6_kernel_m:00@
2assignvariableop_36_adam_conv2d_transpose_6_bias_m:0N
4assignvariableop_37_adam_conv2d_transpose_7_kernel_m: 0@
2assignvariableop_38_adam_conv2d_transpose_7_bias_m: N
4assignvariableop_39_adam_conv2d_transpose_8_kernel_m: @
2assignvariableop_40_adam_conv2d_transpose_8_bias_m:D
*assignvariableop_41_adam_conv2d_6_kernel_v:6
(assignvariableop_42_adam_conv2d_6_bias_v:D
*assignvariableop_43_adam_conv2d_7_kernel_v: 6
(assignvariableop_44_adam_conv2d_7_bias_v: D
*assignvariableop_45_adam_conv2d_8_kernel_v: 06
(assignvariableop_46_adam_conv2d_8_bias_v:0;
)assignvariableop_47_adam_dense_4_kernel_v:0 5
'assignvariableop_48_adam_dense_4_bias_v: ;
)assignvariableop_49_adam_dense_5_kernel_v: 05
'assignvariableop_50_adam_dense_5_bias_v:0N
4assignvariableop_51_adam_conv2d_transpose_6_kernel_v:00@
2assignvariableop_52_adam_conv2d_transpose_6_bias_v:0N
4assignvariableop_53_adam_conv2d_transpose_7_kernel_v: 0@
2assignvariableop_54_adam_conv2d_transpose_7_bias_v: N
4assignvariableop_55_adam_conv2d_transpose_8_kernel_v: @
2assignvariableop_56_adam_conv2d_transpose_8_bias_v:
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_7_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_7_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_8_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_5_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_5_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_6_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_6_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_7_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_7_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_8_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_8_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_6_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_6_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_7_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_7_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_8_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_8_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_4_kernel_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_4_bias_mIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_5_kernel_mIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_5_bias_mIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_conv2d_transpose_6_kernel_mIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_conv2d_transpose_6_bias_mIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_conv2d_transpose_7_kernel_mIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_conv2d_transpose_7_bias_mIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_conv2d_transpose_8_kernel_mIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_conv2d_transpose_8_bias_mIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_6_kernel_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_6_bias_vIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_7_kernel_vIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_7_bias_vIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_8_kernel_vIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_8_bias_vIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_4_kernel_vIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_4_bias_vIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_5_kernel_vIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_5_bias_vIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_6_kernel_vIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_6_bias_vIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_conv2d_transpose_7_kernel_vIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_conv2d_transpose_7_bias_vIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_conv2d_transpose_8_kernel_vIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_conv2d_transpose_8_bias_vIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
)__inference_Decoder_layer_call_fn_3685586
input_8
unknown: 0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3: 0
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685567x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_8
�
G
+__inference_flatten_2_layer_call_fn_3686888

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686135

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
	unknown_7: 0
	unknown_8:0#
	unknown_9:00

unknown_10:0$

unknown_11: 0

unknown_12: $

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:���������@�: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685902x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_6_layer_call_fn_3686802

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�0
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3685029
input_7*
conv2d_6_3684995:
conv2d_6_3684997:*
conv2d_7_3685001: 
conv2d_7_3685003: *
conv2d_8_3685007: 0
conv2d_8_3685009:0!
dense_4_3685014:0 
dense_4_3685016: 
identity

identity_1�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_6_3684995conv2d_6_3684997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_3685001conv2d_7_3685003*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_3685007conv2d_8_3685009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950�
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_3685014dense_4_3685016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976�
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8� *9
f4R2
0__inference_dense_4_activity_regularizer_3684899�
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_7
�	
�
)__inference_Encoder_layer_call_fn_3686439

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3686843

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:��������� _
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
M
1__inference_up_sampling2d_6_layer_call_fn_3687012

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
*__inference_conv2d_7_layer_call_fn_3686832

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
)__inference_Encoder_layer_call_fn_3685148
input_7!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_7
�

�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3686813

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:��������� @_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:��������� @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�	
�
)__inference_Encoder_layer_call_fn_3686461

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685902

inputs)
encoder_3685865:
encoder_3685867:)
encoder_3685869: 
encoder_3685871: )
encoder_3685873: 0
encoder_3685875:0!
encoder_3685877:0 
encoder_3685879: !
decoder_3685883: 0
decoder_3685885:0)
decoder_3685887:00
decoder_3685889:0)
decoder_3685891: 0
decoder_3685893: )
decoder_3685895: 
decoder_3685897:
identity

identity_1��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_3685865encoder_3685867encoder_3685869encoder_3685871encoder_3685873encoder_3685875encoder_3685877encoder_3685879*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685128�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_3685883decoder_3685885decoder_3685887decoder_3685889decoder_3685891decoder_3685893decoder_3685895decoder_3685897*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685617�
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
ވ
�
"__inference__wrapped_model_3684850
input_9c
Iae_conv_prep_flatten_mfcc_encoder_conv2d_6_conv2d_readvariableop_resource:X
Jae_conv_prep_flatten_mfcc_encoder_conv2d_6_biasadd_readvariableop_resource:c
Iae_conv_prep_flatten_mfcc_encoder_conv2d_7_conv2d_readvariableop_resource: X
Jae_conv_prep_flatten_mfcc_encoder_conv2d_7_biasadd_readvariableop_resource: c
Iae_conv_prep_flatten_mfcc_encoder_conv2d_8_conv2d_readvariableop_resource: 0X
Jae_conv_prep_flatten_mfcc_encoder_conv2d_8_biasadd_readvariableop_resource:0Z
Hae_conv_prep_flatten_mfcc_encoder_dense_4_matmul_readvariableop_resource:0 W
Iae_conv_prep_flatten_mfcc_encoder_dense_4_biasadd_readvariableop_resource: Z
Hae_conv_prep_flatten_mfcc_decoder_dense_5_matmul_readvariableop_resource: 0W
Iae_conv_prep_flatten_mfcc_decoder_dense_5_biasadd_readvariableop_resource:0w
]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:00b
Tae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_6_biasadd_readvariableop_resource:0w
]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0b
Tae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_7_biasadd_readvariableop_resource: w
]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: b
Tae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_8_biasadd_readvariableop_resource:
identity��KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp�TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp�KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp�TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp�KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp�TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp�@AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOp�?AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOp�AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOp�@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOp�AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOp�@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOp�AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOp�@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOp�@AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOp�?AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOp�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIae_conv_prep_flatten_mfcc_encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2DConv2Dinput_9HAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_mfcc_encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAddBiasAdd:AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D:output:0IAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
/AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/TanhTanh;AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� @�
9AE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_6/MaxPoolMaxPool3AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOpIae_conv_prep_flatten_mfcc_encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
1AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2DConv2DBAE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_6/MaxPool:output:0HAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_mfcc_encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
2AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAddBiasAdd:AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D:output:0IAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
/AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/TanhTanh;AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
9AE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_7/MaxPoolMaxPool3AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Tanh:y:0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOpIae_conv_prep_flatten_mfcc_encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
1AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2DConv2DBAE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_7/MaxPool:output:0HAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_mfcc_encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
2AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAddBiasAdd:AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D:output:0IAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0�
/AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/TanhTanh;AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
9AE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_8/MaxPoolMaxPool3AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Tanh:y:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
�
1AE_Conv_prep_flatten_MFCC/Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   �
3AE_Conv_prep_flatten_MFCC/Encoder/flatten_2/ReshapeReshapeBAE_Conv_prep_flatten_MFCC/Encoder/max_pooling2d_8/MaxPool:output:0:AE_Conv_prep_flatten_MFCC/Encoder/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������0�
?AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOpReadVariableOpHae_conv_prep_flatten_mfcc_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype0�
0AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMulMatMul<AE_Conv_prep_flatten_MFCC/Encoder/flatten_2/Reshape:output:0GAE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpIae_conv_prep_flatten_mfcc_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
1AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAddBiasAdd:AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul:product:0HAE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.AE_Conv_prep_flatten_MFCC/Encoder/dense_4/TanhTanh:AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
AAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/AbsAbs2AE_Conv_prep_flatten_MFCC/Encoder/dense_4/Tanh:y:0*
T0*'
_output_shapes
:��������� �
CAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
AAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/SumSumEAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/Abs:y:0LAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: �
CAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/mulMulLAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/mul/x:output:0JAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
CAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/ShapeShape2AE_Conv_prep_flatten_MFCC/Encoder/dense_4/Tanh:y:0*
T0*
_output_shapes
::���
QAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
SAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
SAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
KAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSliceLAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/Shape:output:0ZAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0\AE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0\AE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
BAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/CastCastTAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
EAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/truedivRealDivEAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/mul:z:0FAE_Conv_prep_flatten_MFCC/Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
?AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOpReadVariableOpHae_conv_prep_flatten_mfcc_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype0�
0AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMulMatMul2AE_Conv_prep_flatten_MFCC/Encoder/dense_4/Tanh:y:0GAE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
@AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOpIae_conv_prep_flatten_mfcc_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
1AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAddBiasAdd:AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul:product:0HAE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
.AE_Conv_prep_flatten_MFCC/Decoder/dense_5/TanhTanh:AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������0�
1AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/ShapeShape2AE_Conv_prep_flatten_MFCC/Decoder/dense_5/Tanh:y:0*
T0*
_output_shapes
::���
?AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
AAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
AAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_sliceStridedSlice:AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Shape:output:0HAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stack:output:0JAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stack_1:output:0JAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
9AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shapePackBAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/strided_slice:output:0DAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/1:output:0DAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/2:output:0DAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
3AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/ReshapeReshape2AE_Conv_prep_flatten_MFCC/Decoder/dense_5/Tanh:y:0BAE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������0�
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/ShapeShape<AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::���
HAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
BAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_sliceStridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/Shape:output:0QAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stack_1:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0�
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stackPackKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/1:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/2:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
DAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1StridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/stack:output:0\AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0<AE_Conv_prep_flatten_MFCC/Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAddBiasAddNAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0�
9AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/TanhTanhEAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
7AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
9AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
5AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/mulMul@AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/Const:output:0BAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
NAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor=AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/Tanh:y:09AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:���������0*
half_pixel_centers(�
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/ShapeShape_AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::���
HAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
BAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_sliceStridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/Shape:output:0QAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stack_1:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stackPackKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/1:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/2:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
DAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1StridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInputCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/stack:output:0\AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0_AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAddBiasAddNAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
9AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/TanhTanhEAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
7AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
9AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
5AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/mulMul@AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/Const:output:0BAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
NAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/Tanh:y:09AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:���������$$ *
half_pixel_centers(�
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/ShapeShape_AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::���
HAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
BAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_sliceStridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/Shape:output:0QAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stack_1:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :H~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :H~
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
:AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stackPackKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/1:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/2:output:0EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:�
JAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
LAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
DAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1StridedSliceCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stack:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stack_1:output:0UAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp]ae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
EAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transposeConv2DBackpropInputCAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/stack:output:0\AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0_AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:���������HH*
paddingSAME*
strides
�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_mfcc_decoder_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
<AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAddBiasAddNAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose:output:0SAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HH�
9AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/TanhTanhEAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������HH�
7AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"H   H   �
9AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
5AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/mulMul@AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/Const:output:0BAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
NAE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor=AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/Tanh:y:09AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
8AE_Conv_prep_flatten_MFCC/Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
BAE_Conv_prep_flatten_MFCC/Decoder/resizing_2/resize/ResizeBilinearResizeBilinear_AE_Conv_prep_flatten_MFCC/Decoder/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0AAE_Conv_prep_flatten_MFCC/Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(�
IdentityIdentitySAE_Conv_prep_flatten_MFCC/Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:���������@��	
NoOpNoOpL^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpU^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpL^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpU^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpL^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOpU^AE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOpA^AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOp@^AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOpB^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOpB^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOpB^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOpA^AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOp@^AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpTAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpTAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2�
KAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp2�
TAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOpTAE_Conv_prep_flatten_MFCC/Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2�
@AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOp@AE_Conv_prep_flatten_MFCC/Decoder/dense_5/BiasAdd/ReadVariableOp2�
?AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOp?AE_Conv_prep_flatten_MFCC/Decoder/dense_5/MatMul/ReadVariableOp2�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/BiasAdd/ReadVariableOp2�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOp@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_6/Conv2D/ReadVariableOp2�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/BiasAdd/ReadVariableOp2�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOp@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_7/Conv2D/ReadVariableOp2�
AAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/BiasAdd/ReadVariableOp2�
@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOp@AE_Conv_prep_flatten_MFCC/Encoder/conv2d_8/Conv2D/ReadVariableOp2�
@AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOp@AE_Conv_prep_flatten_MFCC/Encoder/dense_4/BiasAdd/ReadVariableOp2�
?AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOp?AE_Conv_prep_flatten_MFCC/Encoder/dense_4/MatMul/ReadVariableOp:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�;
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3686511

inputsA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: A
'conv2d_8_conv2d_readvariableop_resource: 06
(conv2d_8_biasadd_readvariableop_resource:08
&dense_4_matmul_readvariableop_resource:0 5
'dense_4_biasadd_readvariableop_resource: 
identity

identity_1��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @j
conv2d_6/TanhTanhconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� @�
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_7/TanhTanhconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Tanh:y:0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Tanh:y:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   �
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������0�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype0�
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� j
dense_4/ActivityRegularizer/AbsAbsdense_4/Tanh:y:0*
T0*'
_output_shapes
:��������� r
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/Abs:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
!dense_4/ActivityRegularizer/ShapeShapedense_4/Tanh:y:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: _
IdentityIdentitydense_4/Tanh:y:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_8_layer_call_fn_3687093

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_7_layer_call_fn_3687033

inputs!
unknown: 0
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285

inputsB
(conv2d_transpose_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������0j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������0q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������0�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_7_layer_call_fn_3686848

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:���������@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
M
1__inference_up_sampling2d_8_layer_call_fn_3687132

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3687007

inputsB
(conv2d_transpose_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������0j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������0q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������0�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3686883

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�)
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685617

inputs!
dense_5_3685591: 0
dense_5_3685593:04
conv2d_transpose_6_3685597:00(
conv2d_transpose_6_3685599:04
conv2d_transpose_7_3685603: 0(
conv2d_transpose_7_3685605: 4
conv2d_transpose_8_3685609: (
conv2d_transpose_8_3685611:
identity��*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�*conv2d_transpose_8/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_3685591dense_5_3685593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457�
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_6_3685597conv2d_transpose_6_3685599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285�
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_3685603conv2d_transpose_7_3685605*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349�
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372�
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_8_3685609conv2d_transpose_8_3685611*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413�
up_sampling2d_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436�
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_8_layer_call_fn_3686878

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686417

inputsI
/encoder_conv2d_6_conv2d_readvariableop_resource:>
0encoder_conv2d_6_biasadd_readvariableop_resource:I
/encoder_conv2d_7_conv2d_readvariableop_resource: >
0encoder_conv2d_7_biasadd_readvariableop_resource: I
/encoder_conv2d_8_conv2d_readvariableop_resource: 0>
0encoder_conv2d_8_biasadd_readvariableop_resource:0@
.encoder_dense_4_matmul_readvariableop_resource:0 =
/encoder_dense_4_biasadd_readvariableop_resource: @
.decoder_dense_5_matmul_readvariableop_resource: 0=
/decoder_dense_5_biasadd_readvariableop_resource:0]
Cdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:00H
:decoder_conv2d_transpose_6_biasadd_readvariableop_resource:0]
Cdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0H
:decoder_conv2d_transpose_7_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_8_biasadd_readvariableop_resource:
identity

identity_1��1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp�1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp�1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp�:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp�&Decoder/dense_5/BiasAdd/ReadVariableOp�%Decoder/dense_5/MatMul/ReadVariableOp�'Encoder/conv2d_6/BiasAdd/ReadVariableOp�&Encoder/conv2d_6/Conv2D/ReadVariableOp�'Encoder/conv2d_7/BiasAdd/ReadVariableOp�&Encoder/conv2d_7/Conv2D/ReadVariableOp�'Encoder/conv2d_8/BiasAdd/ReadVariableOp�&Encoder/conv2d_8/Conv2D/ReadVariableOp�&Encoder/dense_4/BiasAdd/ReadVariableOp�%Encoder/dense_4/MatMul/ReadVariableOp�
&Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Encoder/conv2d_6/Conv2DConv2Dinputs.Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
'Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/conv2d_6/BiasAddBiasAdd Encoder/conv2d_6/Conv2D:output:0/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @z
Encoder/conv2d_6/TanhTanh!Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� @�
Encoder/max_pooling2d_6/MaxPoolMaxPoolEncoder/conv2d_6/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
&Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Encoder/conv2d_7/Conv2DConv2D(Encoder/max_pooling2d_6/MaxPool:output:0.Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Encoder/conv2d_7/BiasAddBiasAdd Encoder/conv2d_7/Conv2D:output:0/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
Encoder/conv2d_7/TanhTanh!Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
Encoder/max_pooling2d_7/MaxPoolMaxPoolEncoder/conv2d_7/Tanh:y:0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
&Encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Encoder/conv2d_8/Conv2DConv2D(Encoder/max_pooling2d_7/MaxPool:output:0.Encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
'Encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
Encoder/conv2d_8/BiasAddBiasAdd Encoder/conv2d_8/Conv2D:output:0/Encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0z
Encoder/conv2d_8/TanhTanh!Encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
Encoder/max_pooling2d_8/MaxPoolMaxPoolEncoder/conv2d_8/Tanh:y:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   �
Encoder/flatten_2/ReshapeReshape(Encoder/max_pooling2d_8/MaxPool:output:0 Encoder/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������0�
%Encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype0�
Encoder/dense_4/MatMulMatMul"Encoder/flatten_2/Reshape:output:0-Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Encoder/dense_4/BiasAddBiasAdd Encoder/dense_4/MatMul:product:0.Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
Encoder/dense_4/TanhTanh Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� z
'Encoder/dense_4/ActivityRegularizer/AbsAbsEncoder/dense_4/Tanh:y:0*
T0*'
_output_shapes
:��������� z
)Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'Encoder/dense_4/ActivityRegularizer/SumSum+Encoder/dense_4/ActivityRegularizer/Abs:y:02Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'Encoder/dense_4/ActivityRegularizer/mulMul2Encoder/dense_4/ActivityRegularizer/mul/x:output:00Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
)Encoder/dense_4/ActivityRegularizer/ShapeShapeEncoder/dense_4/Tanh:y:0*
T0*
_output_shapes
::���
7Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_4/ActivityRegularizer/Shape:output:0@Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(Encoder/dense_4/ActivityRegularizer/CastCast:Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+Encoder/dense_4/ActivityRegularizer/truedivRealDiv+Encoder/dense_4/ActivityRegularizer/mul:z:0,Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
%Decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype0�
Decoder/dense_5/MatMulMatMulEncoder/dense_4/Tanh:y:0-Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
&Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
Decoder/dense_5/BiasAddBiasAdd Decoder/dense_5/MatMul:product:0.Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0p
Decoder/dense_5/TanhTanh Decoder/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������0m
Decoder/reshape_2/ShapeShapeDecoder/dense_5/Tanh:y:0*
T0*
_output_shapes
::��o
%Decoder/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Decoder/reshape_2/strided_sliceStridedSlice Decoder/reshape_2/Shape:output:0.Decoder/reshape_2/strided_slice/stack:output:00Decoder/reshape_2/strided_slice/stack_1:output:00Decoder/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
Decoder/reshape_2/Reshape/shapePack(Decoder/reshape_2/strided_slice:output:0*Decoder/reshape_2/Reshape/shape/1:output:0*Decoder/reshape_2/Reshape/shape/2:output:0*Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Decoder/reshape_2/ReshapeReshapeDecoder/dense_5/Tanh:y:0(Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������0�
 Decoder/conv2d_transpose_6/ShapeShape"Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_6/strided_sliceStridedSlice)Decoder/conv2d_transpose_6/Shape:output:07Decoder/conv2d_transpose_6/strided_slice/stack:output:09Decoder/conv2d_transpose_6/strided_slice/stack_1:output:09Decoder/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0�
 Decoder/conv2d_transpose_6/stackPack1Decoder/conv2d_transpose_6/strided_slice:output:0+Decoder/conv2d_transpose_6/stack/1:output:0+Decoder/conv2d_transpose_6/stack/2:output:0+Decoder/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_6/strided_slice_1StridedSlice)Decoder/conv2d_transpose_6/stack:output:09Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
+Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_6/stack:output:0BDecoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
"Decoder/conv2d_transpose_6/BiasAddBiasAdd4Decoder/conv2d_transpose_6/conv2d_transpose:output:09Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0�
Decoder/conv2d_transpose_6/TanhTanh+Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������0n
Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_6/mulMul&Decoder/up_sampling2d_6/Const:output:0(Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_6/Tanh:y:0Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:���������0*
half_pixel_centers(�
 Decoder/conv2d_transpose_7/ShapeShapeEDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_7/strided_sliceStridedSlice)Decoder/conv2d_transpose_7/Shape:output:07Decoder/conv2d_transpose_7/strided_slice/stack:output:09Decoder/conv2d_transpose_7/strided_slice/stack_1:output:09Decoder/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
 Decoder/conv2d_transpose_7/stackPack1Decoder/conv2d_transpose_7/strided_slice:output:0+Decoder/conv2d_transpose_7/stack/1:output:0+Decoder/conv2d_transpose_7/stack/2:output:0+Decoder/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_7/strided_slice_1StridedSlice)Decoder/conv2d_transpose_7/stack:output:09Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
+Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_7/stack:output:0BDecoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"Decoder/conv2d_transpose_7/BiasAddBiasAdd4Decoder/conv2d_transpose_7/conv2d_transpose:output:09Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
Decoder/conv2d_transpose_7/TanhTanh+Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� n
Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_7/mulMul&Decoder/up_sampling2d_7/Const:output:0(Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_7/Tanh:y:0Decoder/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:���������$$ *
half_pixel_centers(�
 Decoder/conv2d_transpose_8/ShapeShapeEDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��x
.Decoder/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(Decoder/conv2d_transpose_8/strided_sliceStridedSlice)Decoder/conv2d_transpose_8/Shape:output:07Decoder/conv2d_transpose_8/strided_slice/stack:output:09Decoder/conv2d_transpose_8/strided_slice/stack_1:output:09Decoder/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Hd
"Decoder/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Hd
"Decoder/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
 Decoder/conv2d_transpose_8/stackPack1Decoder/conv2d_transpose_8/strided_slice:output:0+Decoder/conv2d_transpose_8/stack/1:output:0+Decoder/conv2d_transpose_8/stack/2:output:0+Decoder/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*Decoder/conv2d_transpose_8/strided_slice_1StridedSlice)Decoder/conv2d_transpose_8/stack:output:09Decoder/conv2d_transpose_8/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_8/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
+Decoder/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_8/stack:output:0BDecoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:���������HH*
paddingSAME*
strides
�
1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"Decoder/conv2d_transpose_8/BiasAddBiasAdd4Decoder/conv2d_transpose_8/conv2d_transpose:output:09Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HH�
Decoder/conv2d_transpose_8/TanhTanh+Decoder/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������HHn
Decoder/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"H   H   p
Decoder/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Decoder/up_sampling2d_8/mulMul&Decoder/up_sampling2d_8/Const:output:0(Decoder/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
4Decoder/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor#Decoder/conv2d_transpose_8/Tanh:y:0Decoder/up_sampling2d_8/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(o
Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
(Decoder/resizing_2/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(�
IdentityIdentity9Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:���������@�o

Identity_1Identity/Encoder/dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp2^Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp'^Decoder/dense_5/BiasAdd/ReadVariableOp&^Decoder/dense_5/MatMul/ReadVariableOp(^Encoder/conv2d_6/BiasAdd/ReadVariableOp'^Encoder/conv2d_6/Conv2D/ReadVariableOp(^Encoder/conv2d_7/BiasAdd/ReadVariableOp'^Encoder/conv2d_7/Conv2D/ReadVariableOp(^Encoder/conv2d_8/BiasAdd/ReadVariableOp'^Encoder/conv2d_8/Conv2D/ReadVariableOp'^Encoder/dense_4/BiasAdd/ReadVariableOp&^Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2f
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_8/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_5/BiasAdd/ReadVariableOp&Decoder/dense_5/BiasAdd/ReadVariableOp2N
%Decoder/dense_5/MatMul/ReadVariableOp%Decoder/dense_5/MatMul/ReadVariableOp2R
'Encoder/conv2d_6/BiasAdd/ReadVariableOp'Encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_6/Conv2D/ReadVariableOp&Encoder/conv2d_6/Conv2D/ReadVariableOp2R
'Encoder/conv2d_7/BiasAdd/ReadVariableOp'Encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_7/Conv2D/ReadVariableOp&Encoder/conv2d_7/Conv2D/ReadVariableOp2R
'Encoder/conv2d_8/BiasAdd/ReadVariableOp'Encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_8/Conv2D/ReadVariableOp&Encoder/conv2d_8/Conv2D/ReadVariableOp2P
&Encoder/dense_4/BiasAdd/ReadVariableOp&Encoder/dense_4/BiasAdd/ReadVariableOp2N
%Encoder/dense_4/MatMul/ReadVariableOp%Encoder/dense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3686823

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_dense_4_layer_call_and_return_conditional_losses_3686925

inputs0
matmul_readvariableop_resource:0 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:��������� W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������0X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�

�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:��������� @_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:��������� @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_6_layer_call_fn_3686818

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�)
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685567

inputs!
dense_5_3685541: 0
dense_5_3685543:04
conv2d_transpose_6_3685547:00(
conv2d_transpose_6_3685549:04
conv2d_transpose_7_3685553: 0(
conv2d_transpose_7_3685555: 4
conv2d_transpose_8_3685559: (
conv2d_transpose_8_3685561:
identity��*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�*conv2d_transpose_8/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_3685541dense_5_3685543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457�
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_6_3685547conv2d_transpose_6_3685549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285�
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_3685553conv2d_transpose_7_3685555*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349�
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372�
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_8_3685559conv2d_transpose_8_3685561*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3685413�
up_sampling2d_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3685436�
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
)__inference_Decoder_layer_call_fn_3686582

inputs
unknown: 0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3: 0
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685567x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457

inputs0
matmul_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������0W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������0w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685860
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
	unknown_7: 0
	unknown_8:0#
	unknown_9:00

unknown_10:0$

unknown_11: 0

unknown_12: $

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:���������@�: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685824x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3687024

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
G
+__inference_reshape_2_layer_call_fn_3686950

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
M
1__inference_up_sampling2d_7_layer_call_fn_3687072

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3685372�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
)__inference_Encoder_layer_call_fn_3685089
input_7!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_7
�
�
%__inference_signature_wrapper_3686059
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0
	unknown_5:0 
	unknown_6: 
	unknown_7: 0
	unknown_8:0#
	unknown_9:00

unknown_10:0$

unknown_11: 0

unknown_12: $

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_3684850x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�w
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3686793

inputs8
&dense_5_matmul_readvariableop_resource: 05
'dense_5_biasadd_readvariableop_resource:0U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:00@
2conv2d_transpose_6_biasadd_readvariableop_resource:0U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:
identity��)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�)conv2d_transpose_8/BiasAdd/ReadVariableOp�2conv2d_transpose_8/conv2d_transpose/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype0y
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0`
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������0]
reshape_2/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
::��g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_2/ReshapeReshapedense_5/Tanh:y:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������0p
conv2d_transpose_6/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0�
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_2/Reshape:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0~
conv2d_transpose_6/TanhTanh#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������0f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_6/Tanh:y:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:���������0*
half_pixel_centers(�
conv2d_transpose_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_7/Tanh:y:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:���������$$ *
half_pixel_centers(�
conv2d_transpose_8/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��p
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :H\
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :H\
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:���������HH*
paddingSAME*
strides
�
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HH~
conv2d_transpose_8/TanhTanh#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������HHf
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"H   H   h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_8/Tanh:y:0up_sampling2d_8/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(g
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
 resizing_2/resize/ResizeBilinearResizeBilinear=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0resizing_2/resize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(�
IdentityIdentity1resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�5
 __inference__traced_save_3687520
file_prefix@
&read_disablecopyonread_conv2d_6_kernel:4
&read_1_disablecopyonread_conv2d_6_bias:B
(read_2_disablecopyonread_conv2d_7_kernel: 4
&read_3_disablecopyonread_conv2d_7_bias: B
(read_4_disablecopyonread_conv2d_8_kernel: 04
&read_5_disablecopyonread_conv2d_8_bias:09
'read_6_disablecopyonread_dense_4_kernel:0 3
%read_7_disablecopyonread_dense_4_bias: 9
'read_8_disablecopyonread_dense_5_kernel: 03
%read_9_disablecopyonread_dense_5_bias:0M
3read_10_disablecopyonread_conv2d_transpose_6_kernel:00?
1read_11_disablecopyonread_conv2d_transpose_6_bias:0M
3read_12_disablecopyonread_conv2d_transpose_7_kernel: 0?
1read_13_disablecopyonread_conv2d_transpose_7_bias: M
3read_14_disablecopyonread_conv2d_transpose_8_kernel: ?
1read_15_disablecopyonread_conv2d_transpose_8_bias:-
#read_16_disablecopyonread_adam_iter:	 /
%read_17_disablecopyonread_adam_beta_1: /
%read_18_disablecopyonread_adam_beta_2: .
$read_19_disablecopyonread_adam_decay: 6
,read_20_disablecopyonread_adam_learning_rate: +
!read_21_disablecopyonread_total_1: +
!read_22_disablecopyonread_count_1: )
read_23_disablecopyonread_total: )
read_24_disablecopyonread_count: J
0read_25_disablecopyonread_adam_conv2d_6_kernel_m:<
.read_26_disablecopyonread_adam_conv2d_6_bias_m:J
0read_27_disablecopyonread_adam_conv2d_7_kernel_m: <
.read_28_disablecopyonread_adam_conv2d_7_bias_m: J
0read_29_disablecopyonread_adam_conv2d_8_kernel_m: 0<
.read_30_disablecopyonread_adam_conv2d_8_bias_m:0A
/read_31_disablecopyonread_adam_dense_4_kernel_m:0 ;
-read_32_disablecopyonread_adam_dense_4_bias_m: A
/read_33_disablecopyonread_adam_dense_5_kernel_m: 0;
-read_34_disablecopyonread_adam_dense_5_bias_m:0T
:read_35_disablecopyonread_adam_conv2d_transpose_6_kernel_m:00F
8read_36_disablecopyonread_adam_conv2d_transpose_6_bias_m:0T
:read_37_disablecopyonread_adam_conv2d_transpose_7_kernel_m: 0F
8read_38_disablecopyonread_adam_conv2d_transpose_7_bias_m: T
:read_39_disablecopyonread_adam_conv2d_transpose_8_kernel_m: F
8read_40_disablecopyonread_adam_conv2d_transpose_8_bias_m:J
0read_41_disablecopyonread_adam_conv2d_6_kernel_v:<
.read_42_disablecopyonread_adam_conv2d_6_bias_v:J
0read_43_disablecopyonread_adam_conv2d_7_kernel_v: <
.read_44_disablecopyonread_adam_conv2d_7_bias_v: J
0read_45_disablecopyonread_adam_conv2d_8_kernel_v: 0<
.read_46_disablecopyonread_adam_conv2d_8_bias_v:0A
/read_47_disablecopyonread_adam_dense_4_kernel_v:0 ;
-read_48_disablecopyonread_adam_dense_4_bias_v: A
/read_49_disablecopyonread_adam_dense_5_kernel_v: 0;
-read_50_disablecopyonread_adam_dense_5_bias_v:0T
:read_51_disablecopyonread_adam_conv2d_transpose_6_kernel_v:00F
8read_52_disablecopyonread_adam_conv2d_transpose_6_bias_v:0T
:read_53_disablecopyonread_adam_conv2d_transpose_7_kernel_v: 0F
8read_54_disablecopyonread_adam_conv2d_transpose_7_bias_v: T
:read_55_disablecopyonread_adam_conv2d_transpose_8_kernel_v: F
8read_56_disablecopyonread_adam_conv2d_transpose_8_bias_v:
savev2_const
identity_115��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_6_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
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
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_7_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
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
: z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_7_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_8_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_8_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:0{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_4_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0 *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0 e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:0 y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_4_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_5_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 0*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 0e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: 0y
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_5_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_10/DisableCopyOnReadDisableCopyOnRead3read_10_disablecopyonread_conv2d_transpose_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp3read_10_disablecopyonread_conv2d_transpose_6_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:00*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:00m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:00�
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_conv2d_transpose_6_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_conv2d_transpose_6_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_12/DisableCopyOnReadDisableCopyOnRead3read_12_disablecopyonread_conv2d_transpose_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp3read_12_disablecopyonread_conv2d_transpose_7_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0�
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_conv2d_transpose_7_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_conv2d_transpose_7_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead3read_14_disablecopyonread_conv2d_transpose_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp3read_14_disablecopyonread_conv2d_transpose_8_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_conv2d_transpose_8_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_conv2d_transpose_8_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_adam_iter^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_adam_beta_1^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_adam_beta_2^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_adam_decay^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead,read_20_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp,read_20_disablecopyonread_adam_learning_rate^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_total_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_22/DisableCopyOnReadDisableCopyOnRead!read_22_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp!read_22_disablecopyonread_count_1^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_total^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_24/DisableCopyOnReadDisableCopyOnReadread_24_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpread_24_disablecopyonread_count^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_conv2d_6_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_conv2d_6_kernel_m^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_conv2d_6_bias_m"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_conv2d_6_bias_m^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_conv2d_7_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_conv2d_7_kernel_m^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_conv2d_7_bias_m"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_conv2d_7_bias_m^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_conv2d_8_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_conv2d_8_kernel_m^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0�
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_conv2d_8_bias_m"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_conv2d_8_bias_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_dense_4_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_dense_4_kernel_m^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0 *
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0 e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:0 �
Read_32/DisableCopyOnReadDisableCopyOnRead-read_32_disablecopyonread_adam_dense_4_bias_m"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp-read_32_disablecopyonread_adam_dense_4_bias_m^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_adam_dense_5_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_adam_dense_5_kernel_m^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 0*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 0e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

: 0�
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_dense_5_bias_m"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_dense_5_bias_m^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_35/DisableCopyOnReadDisableCopyOnRead:read_35_disablecopyonread_adam_conv2d_transpose_6_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp:read_35_disablecopyonread_adam_conv2d_transpose_6_kernel_m^Read_35/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:00*
dtype0w
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:00m
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*&
_output_shapes
:00�
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_adam_conv2d_transpose_6_bias_m"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_adam_conv2d_transpose_6_bias_m^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_37/DisableCopyOnReadDisableCopyOnRead:read_37_disablecopyonread_adam_conv2d_transpose_7_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp:read_37_disablecopyonread_adam_conv2d_transpose_7_kernel_m^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0�
Read_38/DisableCopyOnReadDisableCopyOnRead8read_38_disablecopyonread_adam_conv2d_transpose_7_bias_m"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp8read_38_disablecopyonread_adam_conv2d_transpose_7_bias_m^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_39/DisableCopyOnReadDisableCopyOnRead:read_39_disablecopyonread_adam_conv2d_transpose_8_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp:read_39_disablecopyonread_adam_conv2d_transpose_8_kernel_m^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_40/DisableCopyOnReadDisableCopyOnRead8read_40_disablecopyonread_adam_conv2d_transpose_8_bias_m"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp8read_40_disablecopyonread_adam_conv2d_transpose_8_bias_m^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_conv2d_6_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_conv2d_6_kernel_v^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_conv2d_6_bias_v"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_conv2d_6_bias_v^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_conv2d_7_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_conv2d_7_kernel_v^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_conv2d_7_bias_v"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_conv2d_7_bias_v^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_conv2d_8_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_conv2d_8_kernel_v^Read_45/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0w
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0m
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0�
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_conv2d_8_bias_v"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_conv2d_8_bias_v^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_47/DisableCopyOnReadDisableCopyOnRead/read_47_disablecopyonread_adam_dense_4_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp/read_47_disablecopyonread_adam_dense_4_kernel_v^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0 *
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0 e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:0 �
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_dense_4_bias_v"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp-read_48_disablecopyonread_adam_dense_4_bias_v^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_dense_5_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_dense_5_kernel_v^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 0*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 0e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

: 0�
Read_50/DisableCopyOnReadDisableCopyOnRead-read_50_disablecopyonread_adam_dense_5_bias_v"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp-read_50_disablecopyonread_adam_dense_5_bias_v^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_51/DisableCopyOnReadDisableCopyOnRead:read_51_disablecopyonread_adam_conv2d_transpose_6_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp:read_51_disablecopyonread_adam_conv2d_transpose_6_kernel_v^Read_51/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:00*
dtype0x
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:00o
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*&
_output_shapes
:00�
Read_52/DisableCopyOnReadDisableCopyOnRead8read_52_disablecopyonread_adam_conv2d_transpose_6_bias_v"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp8read_52_disablecopyonread_adam_conv2d_transpose_6_bias_v^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_53/DisableCopyOnReadDisableCopyOnRead:read_53_disablecopyonread_adam_conv2d_transpose_7_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp:read_53_disablecopyonread_adam_conv2d_transpose_7_kernel_v^Read_53/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0x
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0o
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0�
Read_54/DisableCopyOnReadDisableCopyOnRead8read_54_disablecopyonread_adam_conv2d_transpose_7_bias_v"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp8read_54_disablecopyonread_adam_conv2d_transpose_7_bias_v^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_55/DisableCopyOnReadDisableCopyOnRead:read_55_disablecopyonread_adam_conv2d_transpose_8_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp:read_55_disablecopyonread_adam_conv2d_transpose_8_kernel_v^Read_55/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_56/DisableCopyOnReadDisableCopyOnRead8read_56_disablecopyonread_adam_conv2d_transpose_8_bias_v"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp8read_56_disablecopyonread_adam_conv2d_transpose_8_bias_v^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_114Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_115IdentityIdentity_114:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_115Identity_115:output:0*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp24
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
_user_specified_namefile_prefix::

_output_shapes
: 
�!
�
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3687067

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3687144

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�0
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3685128

inputs*
conv2d_6_3685094:
conv2d_6_3685096:*
conv2d_7_3685100: 
conv2d_7_3685102: *
conv2d_8_3685106: 0
conv2d_8_3685108:0!
dense_4_3685113:0 
dense_4_3685115: 
identity

identity_1�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_3685094conv2d_6_3685096*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3684914�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3684856�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_3685100conv2d_7_3685102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3684868�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_3685106conv2d_8_3685108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950�
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3684963�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_3685113dense_4_3685115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976�
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8� *9
f4R2
0__inference_dense_4_activity_regularizer_3684899�
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
H
,__inference_resizing_2_layer_call_fn_3687149

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_3685503i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
c
G__inference_resizing_2_layer_call_and_return_conditional_losses_3687155

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:���������@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_3685477

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������0`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3687127

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_3686964

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������0`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�

�
D__inference_dense_5_layer_call_and_return_conditional_losses_3686945

inputs0
matmul_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������0W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������0w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3685349

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
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
valueB:�
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
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
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
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�

�
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976

inputs0
matmul_readvariableop_resource:0 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:��������� W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
)__inference_dense_4_layer_call_fn_3686903

inputs
unknown:0 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�

�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3684932

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:��������� _
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3684880

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
)__inference_Decoder_layer_call_fn_3686603

inputs
unknown: 0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3: 0
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685617x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685741
input_9)
encoder_3685704:
encoder_3685706:)
encoder_3685708: 
encoder_3685710: )
encoder_3685712: 0
encoder_3685714:0!
encoder_3685716:0 
encoder_3685718: !
decoder_3685722: 0
decoder_3685724:0)
decoder_3685726:00
decoder_3685728:0)
decoder_3685730: 0
decoder_3685732: )
decoder_3685734: 
decoder_3685736:
identity

identity_1��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_3685704encoder_3685706encoder_3685708encoder_3685710encoder_3685712encoder_3685714encoder_3685716encoder_3685718*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:��������� : **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_3685069�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_3685722decoder_3685724decoder_3685726decoder_3685728decoder_3685730decoder_3685732decoder_3685734decoder_3685736*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685567�
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������@�: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:���������@�
!
_user_specified_name	input_9
�	
�
)__inference_Decoder_layer_call_fn_3685636
input_8
unknown: 0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3: 0
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@�**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_3685617x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_8
�

�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3684950

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������0_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:���������0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3686853

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3686873

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������0_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:���������0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
H__inference_dense_4_layer_call_and_return_all_conditional_losses_3686914

inputs
unknown:0 
	unknown_0: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3684976�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8� *9
f4R2
0__inference_dense_4_activity_regularizer_3684899o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_6_layer_call_fn_3686973

inputs!
unknown:00
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3685285�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������0
 
_user_specified_nameinputs
�;
�
D__inference_Encoder_layer_call_and_return_conditional_losses_3686561

inputsA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: A
'conv2d_8_conv2d_readvariableop_resource: 06
(conv2d_8_biasadd_readvariableop_resource:08
&dense_4_matmul_readvariableop_resource:0 5
'dense_4_biasadd_readvariableop_resource: 
identity

identity_1��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @j
conv2d_6/TanhTanhconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� @�
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_7/TanhTanhconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Tanh:y:0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Tanh:y:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����0   �
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������0�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype0�
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� j
dense_4/ActivityRegularizer/AbsAbsdense_4/Tanh:y:0*
T0*'
_output_shapes
:��������� r
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/Abs:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
!dense_4/ActivityRegularizer/ShapeShapedense_4/Tanh:y:0*
T0*
_output_shapes
::��y
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: _
IdentityIdentitydense_4/Tanh:y:0^NoOp*
T0*'
_output_shapes
:��������� g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������@�: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������@�
 
_user_specified_nameinputs
�
G
0__inference_dense_4_activity_regularizer_3684899
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
�
)__inference_dense_5_layer_call_fn_3686934

inputs
unknown: 0
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_3685457o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�w
�
D__inference_Decoder_layer_call_and_return_conditional_losses_3686698

inputs8
&dense_5_matmul_readvariableop_resource: 05
'dense_5_biasadd_readvariableop_resource:0U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:00@
2conv2d_transpose_6_biasadd_readvariableop_resource:0U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:
identity��)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�)conv2d_transpose_8/BiasAdd/ReadVariableOp�2conv2d_transpose_8/conv2d_transpose/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype0y
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0`
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������0]
reshape_2/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
::��g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_2/ReshapeReshapedense_5/Tanh:y:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������0p
conv2d_transpose_6/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0�
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:00*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_2/Reshape:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0~
conv2d_transpose_6/TanhTanh#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������0f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_6/Tanh:y:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:���������0*
half_pixel_centers(�
conv2d_transpose_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:��������� f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_7/Tanh:y:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:���������$$ *
half_pixel_centers(�
conv2d_transpose_8/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::��p
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :H\
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :H\
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:���������HH*
paddingSAME*
strides
�
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HH~
conv2d_transpose_8/TanhTanh#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������HHf
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"H   H   h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_transpose_8/Tanh:y:0up_sampling2d_8/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(g
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   �   �
 resizing_2/resize/ResizeBilinearResizeBilinear=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0resizing_2/resize/size:output:0*
T0*0
_output_shapes
:���������@�*
half_pixel_centers(�
IdentityIdentity1resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:���������@��
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : : : : : : : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3685308

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_99
serving_default_input_9:0���������@�D
Decoder9
StatefulPartitionedCall:0���������@�tensorflow/serving/predict:��
�
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
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_sequential
�
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"layer_with_weights-2
"layer-4
#layer-5
$layer_with_weights-3
$layer-6
%layer-7
&layer-8
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
#-_self_saveable_object_factories"
_tf_keras_sequential
�
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15"
trackable_list_wrapper
�
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685860
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685938
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686097
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686135�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
�
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685741
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685781
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686276
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686417�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
�B�
"__inference__wrapped_model_3684850input_9"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_rate.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�<m�=m�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�<v�=v�"
	optimizer
,
Pserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

.kernel
/bias
#W_self_saveable_object_factories
 X_jit_compiled_convolution_op"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

0kernel
1bias
#f_self_saveable_object_factories
 g_jit_compiled_convolution_op"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
#n_self_saveable_object_factories"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

2kernel
3bias
#u_self_saveable_object_factories
 v_jit_compiled_convolution_op"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
#}_self_saveable_object_factories"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

4kernel
5bias
$�_self_saveable_object_factories"
_tf_keras_layer
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_Encoder_layer_call_fn_3685089
)__inference_Encoder_layer_call_fn_3685148
)__inference_Encoder_layer_call_fn_3686439
)__inference_Encoder_layer_call_fn_3686461�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_Encoder_layer_call_and_return_conditional_losses_3684992
D__inference_Encoder_layer_call_and_return_conditional_losses_3685029
D__inference_Encoder_layer_call_and_return_conditional_losses_3686511
D__inference_Encoder_layer_call_and_return_conditional_losses_3686561�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

6kernel
7bias
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

8kernel
9bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

:kernel
;bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

<kernel
=bias
$�_self_saveable_object_factories
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
X
60
71
82
93
:4
;5
<6
=7"
trackable_list_wrapper
X
60
71
82
93
:4
;5
<6
=7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_Decoder_layer_call_fn_3685586
)__inference_Decoder_layer_call_fn_3685636
)__inference_Decoder_layer_call_fn_3686582
)__inference_Decoder_layer_call_fn_3686603�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685506
D__inference_Decoder_layer_call_and_return_conditional_losses_3685535
D__inference_Decoder_layer_call_and_return_conditional_losses_3686698
D__inference_Decoder_layer_call_and_return_conditional_losses_3686793�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
 "
trackable_dict_wrapper
):'2conv2d_6/kernel
:2conv2d_6/bias
):' 2conv2d_7/kernel
: 2conv2d_7/bias
):' 02conv2d_8/kernel
:02conv2d_8/bias
 :0 2dense_4/kernel
: 2dense_4/bias
 : 02dense_5/kernel
:02dense_5/bias
3:1002conv2d_transpose_6/kernel
%:#02conv2d_transpose_6/bias
3:1 02conv2d_transpose_7/kernel
%:# 2conv2d_transpose_7/bias
3:1 2conv2d_transpose_8/kernel
%:#2conv2d_transpose_8/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685860input_9"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685938input_9"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686097inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686135inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685741input_9"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685781input_9"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686276inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686417inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_3686059input_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_6_layer_call_fn_3686802�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3686813�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_6_layer_call_fn_3686818�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3686823�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_7_layer_call_fn_3686832�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3686843�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_7_layer_call_fn_3686848�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3686853�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_8_layer_call_fn_3686862�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3686873�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_8_layer_call_fn_3686878�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3686883�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_2_layer_call_fn_3686888�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_2_layer_call_and_return_conditional_losses_3686894�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
�activity_regularizer_fn
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_4_layer_call_fn_3686903�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_dense_4_layer_call_and_return_all_conditional_losses_3686914�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_Encoder_layer_call_fn_3685089input_7"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Encoder_layer_call_fn_3685148input_7"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Encoder_layer_call_fn_3686439inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Encoder_layer_call_fn_3686461inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Encoder_layer_call_and_return_conditional_losses_3684992input_7"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Encoder_layer_call_and_return_conditional_losses_3685029input_7"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Encoder_layer_call_and_return_conditional_losses_3686511inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Encoder_layer_call_and_return_conditional_losses_3686561inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_5_layer_call_fn_3686934�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_5_layer_call_and_return_conditional_losses_3686945�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_reshape_2_layer_call_fn_3686950�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_reshape_2_layer_call_and_return_conditional_losses_3686964�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_conv2d_transpose_6_layer_call_fn_3686973�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3687007�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_6_layer_call_fn_3687012�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3687024�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_conv2d_transpose_7_layer_call_fn_3687033�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3687067�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_7_layer_call_fn_3687072�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3687084�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_conv2d_transpose_8_layer_call_fn_3687093�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3687127�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_8_layer_call_fn_3687132�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3687144�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_resizing_2_layer_call_fn_3687149�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_resizing_2_layer_call_and_return_conditional_losses_3687155�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
 2
!3
"4
#5
$6
%7
&8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_Decoder_layer_call_fn_3685586input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Decoder_layer_call_fn_3685636input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Decoder_layer_call_fn_3686582inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Decoder_layer_call_fn_3686603inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685506input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Decoder_layer_call_and_return_conditional_losses_3685535input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Decoder_layer_call_and_return_conditional_losses_3686698inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Decoder_layer_call_and_return_conditional_losses_3686793inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
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
�B�
*__inference_conv2d_6_layer_call_fn_3686802inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3686813inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_max_pooling2d_6_layer_call_fn_3686818inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3686823inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_7_layer_call_fn_3686832inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3686843inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_max_pooling2d_7_layer_call_fn_3686848inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3686853inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_8_layer_call_fn_3686862inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3686873inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_max_pooling2d_8_layer_call_fn_3686878inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3686883inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_flatten_2_layer_call_fn_3686888inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_2_layer_call_and_return_conditional_losses_3686894inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
�trace_02�
0__inference_dense_4_activity_regularizer_3684899�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�z�trace_0
�
�trace_02�
D__inference_dense_4_layer_call_and_return_conditional_losses_3686925�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
)__inference_dense_4_layer_call_fn_3686903inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4_layer_call_and_return_all_conditional_losses_3686914inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_dense_5_layer_call_fn_3686934inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_5_layer_call_and_return_conditional_losses_3686945inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_reshape_2_layer_call_fn_3686950inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_reshape_2_layer_call_and_return_conditional_losses_3686964inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
4__inference_conv2d_transpose_6_layer_call_fn_3686973inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3687007inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_up_sampling2d_6_layer_call_fn_3687012inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3687024inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
4__inference_conv2d_transpose_7_layer_call_fn_3687033inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3687067inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_up_sampling2d_7_layer_call_fn_3687072inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3687084inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
4__inference_conv2d_transpose_8_layer_call_fn_3687093inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3687127inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_up_sampling2d_8_layer_call_fn_3687132inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3687144inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_resizing_2_layer_call_fn_3687149inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_resizing_2_layer_call_and_return_conditional_losses_3687155inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�B�
0__inference_dense_4_activity_regularizer_3684899x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
D__inference_dense_4_layer_call_and_return_conditional_losses_3686925inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
.:, 2Adam/conv2d_7/kernel/m
 : 2Adam/conv2d_7/bias/m
.:, 02Adam/conv2d_8/kernel/m
 :02Adam/conv2d_8/bias/m
%:#0 2Adam/dense_4/kernel/m
: 2Adam/dense_4/bias/m
%:# 02Adam/dense_5/kernel/m
:02Adam/dense_5/bias/m
8:6002 Adam/conv2d_transpose_6/kernel/m
*:(02Adam/conv2d_transpose_6/bias/m
8:6 02 Adam/conv2d_transpose_7/kernel/m
*:( 2Adam/conv2d_transpose_7/bias/m
8:6 2 Adam/conv2d_transpose_8/kernel/m
*:(2Adam/conv2d_transpose_8/bias/m
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
.:, 2Adam/conv2d_7/kernel/v
 : 2Adam/conv2d_7/bias/v
.:, 02Adam/conv2d_8/kernel/v
 :02Adam/conv2d_8/bias/v
%:#0 2Adam/dense_4/kernel/v
: 2Adam/dense_4/bias/v
%:# 02Adam/dense_5/kernel/v
:02Adam/dense_5/bias/v
8:6002 Adam/conv2d_transpose_6/kernel/v
*:(02Adam/conv2d_transpose_6/bias/v
8:6 02 Adam/conv2d_transpose_7/kernel/v
*:( 2Adam/conv2d_transpose_7/bias/v
8:6 2 Adam/conv2d_transpose_8/kernel/v
*:(2Adam/conv2d_transpose_8/bias/v�
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685741�./0123456789:;<=A�>
7�4
*�'
input_9���������@�
p

 
� "J�G
+�(
tensor_0���������@�
�
�

tensor_1_0 �
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3685781�./0123456789:;<=A�>
7�4
*�'
input_9���������@�
p 

 
� "J�G
+�(
tensor_0���������@�
�
�

tensor_1_0 �
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686276�./0123456789:;<=@�=
6�3
)�&
inputs���������@�
p

 
� "J�G
+�(
tensor_0���������@�
�
�

tensor_1_0 �
V__inference_AE_Conv_prep_flatten_MFCC_layer_call_and_return_conditional_losses_3686417�./0123456789:;<=@�=
6�3
)�&
inputs���������@�
p 

 
� "J�G
+�(
tensor_0���������@�
�
�

tensor_1_0 �
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685860�./0123456789:;<=A�>
7�4
*�'
input_9���������@�
p

 
� "*�'
unknown���������@��
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3685938�./0123456789:;<=A�>
7�4
*�'
input_9���������@�
p 

 
� "*�'
unknown���������@��
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686097�./0123456789:;<=@�=
6�3
)�&
inputs���������@�
p

 
� "*�'
unknown���������@��
;__inference_AE_Conv_prep_flatten_MFCC_layer_call_fn_3686135�./0123456789:;<=@�=
6�3
)�&
inputs���������@�
p 

 
� "*�'
unknown���������@��
D__inference_Decoder_layer_call_and_return_conditional_losses_3685506{6789:;<=8�5
.�+
!�
input_8��������� 
p

 
� "5�2
+�(
tensor_0���������@�
� �
D__inference_Decoder_layer_call_and_return_conditional_losses_3685535{6789:;<=8�5
.�+
!�
input_8��������� 
p 

 
� "5�2
+�(
tensor_0���������@�
� �
D__inference_Decoder_layer_call_and_return_conditional_losses_3686698z6789:;<=7�4
-�*
 �
inputs��������� 
p

 
� "5�2
+�(
tensor_0���������@�
� �
D__inference_Decoder_layer_call_and_return_conditional_losses_3686793z6789:;<=7�4
-�*
 �
inputs��������� 
p 

 
� "5�2
+�(
tensor_0���������@�
� �
)__inference_Decoder_layer_call_fn_3685586p6789:;<=8�5
.�+
!�
input_8��������� 
p

 
� "*�'
unknown���������@��
)__inference_Decoder_layer_call_fn_3685636p6789:;<=8�5
.�+
!�
input_8��������� 
p 

 
� "*�'
unknown���������@��
)__inference_Decoder_layer_call_fn_3686582o6789:;<=7�4
-�*
 �
inputs��������� 
p

 
� "*�'
unknown���������@��
)__inference_Decoder_layer_call_fn_3686603o6789:;<=7�4
-�*
 �
inputs��������� 
p 

 
� "*�'
unknown���������@��
D__inference_Encoder_layer_call_and_return_conditional_losses_3684992�./012345A�>
7�4
*�'
input_7���������@�
p

 
� "A�>
"�
tensor_0��������� 
�
�

tensor_1_0 �
D__inference_Encoder_layer_call_and_return_conditional_losses_3685029�./012345A�>
7�4
*�'
input_7���������@�
p 

 
� "A�>
"�
tensor_0��������� 
�
�

tensor_1_0 �
D__inference_Encoder_layer_call_and_return_conditional_losses_3686511�./012345@�=
6�3
)�&
inputs���������@�
p

 
� "A�>
"�
tensor_0��������� 
�
�

tensor_1_0 �
D__inference_Encoder_layer_call_and_return_conditional_losses_3686561�./012345@�=
6�3
)�&
inputs���������@�
p 

 
� "A�>
"�
tensor_0��������� 
�
�

tensor_1_0 �
)__inference_Encoder_layer_call_fn_3685089p./012345A�>
7�4
*�'
input_7���������@�
p

 
� "!�
unknown��������� �
)__inference_Encoder_layer_call_fn_3685148p./012345A�>
7�4
*�'
input_7���������@�
p 

 
� "!�
unknown��������� �
)__inference_Encoder_layer_call_fn_3686439o./012345@�=
6�3
)�&
inputs���������@�
p

 
� "!�
unknown��������� �
)__inference_Encoder_layer_call_fn_3686461o./012345@�=
6�3
)�&
inputs���������@�
p 

 
� "!�
unknown��������� �
"__inference__wrapped_model_3684850�./0123456789:;<=9�6
/�,
*�'
input_9���������@�
� ":�7
5
Decoder*�'
decoder���������@��
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3686813t./8�5
.�+
)�&
inputs���������@�
� "4�1
*�'
tensor_0��������� @
� �
*__inference_conv2d_6_layer_call_fn_3686802i./8�5
.�+
)�&
inputs���������@�
� ")�&
unknown��������� @�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3686843s017�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0��������� 
� �
*__inference_conv2d_7_layer_call_fn_3686832h017�4
-�*
(�%
inputs���������
� ")�&
unknown��������� �
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3686873s237�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������0
� �
*__inference_conv2d_8_layer_call_fn_3686862h237�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������0�
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_3687007�89I�F
?�<
:�7
inputs+���������������������������0
� "F�C
<�9
tensor_0+���������������������������0
� �
4__inference_conv2d_transpose_6_layer_call_fn_3686973�89I�F
?�<
:�7
inputs+���������������������������0
� ";�8
unknown+���������������������������0�
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_3687067�:;I�F
?�<
:�7
inputs+���������������������������0
� "F�C
<�9
tensor_0+��������������������������� 
� �
4__inference_conv2d_transpose_7_layer_call_fn_3687033�:;I�F
?�<
:�7
inputs+���������������������������0
� ";�8
unknown+��������������������������� �
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_3687127�<=I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
4__inference_conv2d_transpose_8_layer_call_fn_3687093�<=I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+���������������������������c
0__inference_dense_4_activity_regularizer_3684899/�
�
�	
x
� "�
unknown �
H__inference_dense_4_layer_call_and_return_all_conditional_losses_3686914x45/�,
%�"
 �
inputs���������0
� "A�>
"�
tensor_0��������� 
�
�

tensor_1_0 �
D__inference_dense_4_layer_call_and_return_conditional_losses_3686925c45/�,
%�"
 �
inputs���������0
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_4_layer_call_fn_3686903X45/�,
%�"
 �
inputs���������0
� "!�
unknown��������� �
D__inference_dense_5_layer_call_and_return_conditional_losses_3686945c67/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������0
� �
)__inference_dense_5_layer_call_fn_3686934X67/�,
%�"
 �
inputs��������� 
� "!�
unknown���������0�
F__inference_flatten_2_layer_call_and_return_conditional_losses_3686894g7�4
-�*
(�%
inputs���������0
� ",�)
"�
tensor_0���������0
� �
+__inference_flatten_2_layer_call_fn_3686888\7�4
-�*
(�%
inputs���������0
� "!�
unknown���������0�
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3686823�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_6_layer_call_fn_3686818�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3686853�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_7_layer_call_fn_3686848�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3686883�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_8_layer_call_fn_3686878�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
F__inference_reshape_2_layer_call_and_return_conditional_losses_3686964g/�,
%�"
 �
inputs���������0
� "4�1
*�'
tensor_0���������0
� �
+__inference_reshape_2_layer_call_fn_3686950\/�,
%�"
 �
inputs���������0
� ")�&
unknown���������0�
G__inference_resizing_2_layer_call_and_return_conditional_losses_3687155�I�F
?�<
:�7
inputs+���������������������������
� "5�2
+�(
tensor_0���������@�
� �
,__inference_resizing_2_layer_call_fn_3687149wI�F
?�<
:�7
inputs+���������������������������
� "*�'
unknown���������@��
%__inference_signature_wrapper_3686059�./0123456789:;<=D�A
� 
:�7
5
input_9*�'
input_9���������@�":�7
5
Decoder*�'
decoder���������@��
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_3687024�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_6_layer_call_fn_3687012�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_3687084�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_7_layer_call_fn_3687072�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_3687144�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_8_layer_call_fn_3687132�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������