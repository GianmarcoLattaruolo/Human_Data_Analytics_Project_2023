єњ
мѕ
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
validate_shapebool( И
А
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
Ы
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
ј
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
resourceИ
;
Elu
features"T
activations"T"
Ttype:
2
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
В
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
Т
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
Щ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
output"out_typeКнout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
О
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02unknown8ез
Ф
Adam/conv2d_transpose_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_7/bias/v
Н
2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/v*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_7/kernel/v
Э
4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/v*&
_output_shapes
:@*
dtype0
Ф
Adam/conv2d_transpose_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_6/bias/v
Н
2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/v*
_output_shapes
:@*
dtype0
§
 Adam/conv2d_transpose_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*1
shared_name" Adam/conv2d_transpose_6/kernel/v
Э
4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/v*&
_output_shapes
:@`*
dtype0
Ф
Adam/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*/
shared_name Adam/conv2d_transpose_5/bias/v
Н
2Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/v*
_output_shapes
:`*
dtype0
§
 Adam/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:``*1
shared_name" Adam/conv2d_transpose_5/kernel/v
Э
4Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/v*&
_output_shapes
:``*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes	
:ј*
dtype0
З
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ј*&
shared_nameAdam/dense_5/kernel/v
А
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	 ј*
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
З
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј *&
shared_nameAdam/dense_4/kernel/v
А
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	ј *
dtype0
А
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:`*
dtype0
Р
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*'
shared_nameAdam/conv2d_7/kernel/v
Й
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:@`*
dtype0
А
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_6/kernel/v
Й
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/v
Й
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
: *
dtype0
Ф
Adam/conv2d_transpose_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_7/bias/m
Н
2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/m*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_7/kernel/m
Э
4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/m*&
_output_shapes
:@*
dtype0
Ф
Adam/conv2d_transpose_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_6/bias/m
Н
2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/m*
_output_shapes
:@*
dtype0
§
 Adam/conv2d_transpose_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*1
shared_name" Adam/conv2d_transpose_6/kernel/m
Э
4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/m*&
_output_shapes
:@`*
dtype0
Ф
Adam/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*/
shared_name Adam/conv2d_transpose_5/bias/m
Н
2Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/m*
_output_shapes
:`*
dtype0
§
 Adam/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:``*1
shared_name" Adam/conv2d_transpose_5/kernel/m
Э
4Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/m*&
_output_shapes
:``*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:ј*
dtype0
З
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ј*&
shared_nameAdam/dense_5/kernel/m
А
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	 ј*
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
З
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј *&
shared_nameAdam/dense_4/kernel/m
А
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	ј *
dtype0
А
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:`*
dtype0
Р
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*'
shared_nameAdam/conv2d_7/kernel/m
Й
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:@`*
dtype0
А
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_6/kernel/m
Й
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/m
Й
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
: *
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
Ж
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
:*
dtype0
Ц
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_7/kernel
П
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
:@*
dtype0
Ж
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
:@*
dtype0
Ц
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`**
shared_nameconv2d_transpose_6/kernel
П
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
:@`*
dtype0
Ж
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:`*
dtype0
Ц
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:``**
shared_nameconv2d_transpose_5/kernel
П
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:``*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:ј*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ј*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	 ј*
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
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј *
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	ј *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:`*
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@`*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
М
serving_default_input_9Placeholder*0
_output_shapes
:€€€€€€€€€@А*
dtype0*%
shape:€€€€€€€€€@А
Ф
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2272728

NoOpNoOp
ћђ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Жђ
valueыЂBчЂ BпЂ
ћ
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
≠
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
# _self_saveable_object_factories*
Т
!layer_with_weights-0
!layer-0
"layer-1
#layer_with_weights-1
#layer-2
$layer-3
%layer_with_weights-2
%layer-4
&layer-5
'layer_with_weights-3
'layer-6
(layer-7
)layer-8
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
#0_self_saveable_object_factories*
z
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15*
z
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15*
* 
∞
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
* 
Д
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_rate1mІ2m®3m©4m™5mЂ6mђ7m≠8mЃ9mѓ:m∞;m±<m≤=m≥>mі?mµ@mґ1vЈ2vЄ3vє4vЇ5vї6vЉ7vљ8vЊ9vњ:vј;vЅ<v¬=v√>vƒ?v≈@v∆*

Sserving_default* 
* 
* 
н
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

1kernel
2bias
#Z_self_saveable_object_factories
 [_jit_compiled_convolution_op*
≥
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
#b_self_saveable_object_factories* 
 
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
i_random_generator
#j_self_saveable_object_factories* 
н
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

3kernel
4bias
#q_self_saveable_object_factories
 r_jit_compiled_convolution_op*
≥
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories* 
ћ
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
А_random_generator
$Б_self_saveable_object_factories* 
х
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses

5kernel
6bias
$И_self_saveable_object_factories
!Й_jit_compiled_convolution_op*
Ї
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses
$Р_self_saveable_object_factories* 
“
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator
$Ш_self_saveable_object_factories* 
Ї
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
$Я_self_saveable_object_factories* 
“
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses

7kernel
8bias
$¶_self_saveable_object_factories*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 
Ш
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
ђtrace_0
≠trace_1
Ѓtrace_2
ѓtrace_3* 
:
∞trace_0
±trace_1
≤trace_2
≥trace_3* 
* 
“
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
Є__call__
+є&call_and_return_all_conditional_losses

9kernel
:bias
$Ї_self_saveable_object_factories*
Ї
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses
$Ѕ_self_saveable_object_factories* 
х
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses

;kernel
<bias
$»_self_saveable_object_factories
!…_jit_compiled_convolution_op*
Ї
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
$–_self_saveable_object_factories* 
х
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses

=kernel
>bias
$„_self_saveable_object_factories
!Ў_jit_compiled_convolution_op*
Ї
ў	variables
Џtrainable_variables
џregularization_losses
№	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses
$я_self_saveable_object_factories* 
х
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

?kernel
@bias
$ж_self_saveable_object_factories
!з_jit_compiled_convolution_op*
Ї
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses
$о_self_saveable_object_factories* 
Ї
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
$х_self_saveable_object_factories* 
<
90
:1
;2
<3
=4
>5
?6
@7*
<
90
:1
;2
<3
=4
>5
?6
@7*
* 
Ш
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
:
ыtrace_0
ьtrace_1
эtrace_2
юtrace_3* 
:
€trace_0
Аtrace_1
Бtrace_2
Вtrace_3* 
* 
OI
VARIABLE_VALUEconv2d_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_6/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_7/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_7/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

Г0
Д1*
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
10
21*

10
21*
* 
Ш
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
* 
* 
* 
* 
* 
Ц
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
* 
* 
* 
* 
Ц
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
(
$Ь_self_saveable_object_factories* 
* 

30
41*

30
41*
* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

Ґtrace_0* 

£trace_0* 
* 
* 
* 
* 
* 
Ц
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

©trace_0* 

™trace_0* 
* 
* 
* 
* 
Ц
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

∞trace_0
±trace_1* 

≤trace_0
≥trace_1* 
(
$і_self_saveable_object_factories* 
* 

50
61*

50
61*
* 
Ю
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses*

Їtrace_0* 

їtrace_0* 
* 
* 
* 
* 
* 
Ь
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 

Ѕtrace_0* 

¬trace_0* 
* 
* 
* 
* 
Ь
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

»trace_0
…trace_1* 

 trace_0
Ћtrace_1* 
(
$ћ_self_saveable_object_factories* 
* 
* 
* 
* 
Ь
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

“trace_0* 

”trace_0* 
* 

70
81*

70
81*
* 
Љ
‘non_trainable_variables
’layers
÷metrics
 „layer_regularization_losses
Ўlayer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
ўactivity_regularizer_fn
+•&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses*

џtrace_0* 

№trace_0* 
* 
* 
R
0
1
2
3
4
5
6
7
8
9
10*
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
90
:1*

90
:1*
* 
Ю
Ёnon_trainable_variables
ёlayers
яmetrics
 аlayer_regularization_losses
бlayer_metrics
і	variables
µtrainable_variables
ґregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses*

вtrace_0* 

гtrace_0* 
* 
* 
* 
* 
Ь
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses* 

йtrace_0* 

кtrace_0* 
* 

;0
<1*

;0
<1*
* 
Ю
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
* 
* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 
* 

=0
>1*

=0
>1*
* 
Ю
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*

юtrace_0* 

€trace_0* 
* 
* 
* 
* 
* 
Ь
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
ў	variables
Џtrainable_variables
џregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 
* 

?0
@1*

?0
@1*
* 
Ю
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
* 
* 
* 
* 
* 
Ь
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses* 

Уtrace_0* 

Фtrace_0* 
* 
* 
* 
* 
Ь
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 
* 
* 
C
!0
"1
#2
$3
%4
&5
'6
(7
)8*
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
Ь	variables
Э	keras_api

Юtotal

Яcount*
M
†	variables
°	keras_api

Ґtotal

£count
§
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
•trace_0* 

¶trace_0* 
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
Ю0
Я1*

Ь	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ґ0
£1*

†	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_6/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_6/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_7/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_7/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_4/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_4/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_5/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_6/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_6/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_7/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_7/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_4/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_4/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_5/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¬
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/conv2d_transpose_5/kernel/mAdam/conv2d_transpose_5/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/conv2d_transpose_5/kernel/vAdam/conv2d_transpose_5/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/vConst*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_2274324
љ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/conv2d_transpose_5/kernel/mAdam/conv2d_transpose_5/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/conv2d_transpose_5/kernel/vAdam/conv2d_transpose_5/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_2274505І¶
ƒ<
П
D__inference_Encoder_layer_call_and_return_conditional_losses_2271702

inputs*
conv2d_5_2271665: 
conv2d_5_2271667: *
conv2d_6_2271672: @
conv2d_6_2271674:@*
conv2d_7_2271679:@`
conv2d_7_2271681:`"
dense_4_2271687:	ј 
dense_4_2271689: 
identity

identity_1ИҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallы
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_2271665conv2d_5_2271667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ @ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484т
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426с
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271503Э
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_6_2271672conv2d_6_2271674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516т
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438Ч
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271535Я
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_7_2271679conv2d_7_2271681*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548т
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450Щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271567а
flatten_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575Л
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_2271687dense_4_2271689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588»
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
 *-
config_proto

CPU

GPU 2J 8В *9
f4R2
0__inference_dense_4_activity_regularizer_2271469З
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ђ
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ї
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
ё

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273625

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«<
Р
D__inference_Encoder_layer_call_and_return_conditional_losses_2271604
input_7*
conv2d_5_2271485: 
conv2d_5_2271487: *
conv2d_6_2271517: @
conv2d_6_2271519:@*
conv2d_7_2271549:@`
conv2d_7_2271551:`"
dense_4_2271589:	ј 
dense_4_2271591: 
identity

identity_1ИҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallь
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_5_2271485conv2d_5_2271487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ @ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484т
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426с
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271503Э
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_6_2271517conv2d_6_2271519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516т
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438Ч
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271535Я
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_7_2271549conv2d_7_2271551*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548т
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450Щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271567а
flatten_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575Л
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_2271589dense_4_2271591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588»
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
 *-
config_proto

CPU

GPU 2J 8В *9
f4R2
0__inference_dense_4_activity_regularizer_2271469З
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ђ
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ї
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_7
†

ч
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126

inputs1
matmul_readvariableop_resource:	 ј.
biasadd_readvariableop_resource:	ј
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јO
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јa
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€јw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
…
©
4__inference_conv2d_transpose_6_layer_call_fn_2273837

inputs!
unknown:@`
	unknown_0:@
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
ч
b
D__inference_dropout_layer_call_and_return_conditional_losses_2273573

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
э
G
0__inference_dense_4_activity_regularizer_2271469
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
:€€€€€€€€€D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:I
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
Ь

ц
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588

inputs1
matmul_readvariableop_resource:	ј -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Б
ю
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
У
h
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н	
”
)__inference_Decoder_layer_call_fn_2272305
input_8
unknown:	 ј
	unknown_0:	ј#
	unknown_1:``
	unknown_2:`#
	unknown_3:@`
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272286x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€ 
!
_user_specified_name	input_8
ПЭ
¬5
 __inference__traced_save_2274324
file_prefix@
&read_disablecopyonread_conv2d_5_kernel: 4
&read_1_disablecopyonread_conv2d_5_bias: B
(read_2_disablecopyonread_conv2d_6_kernel: @4
&read_3_disablecopyonread_conv2d_6_bias:@B
(read_4_disablecopyonread_conv2d_7_kernel:@`4
&read_5_disablecopyonread_conv2d_7_bias:`:
'read_6_disablecopyonread_dense_4_kernel:	ј 3
%read_7_disablecopyonread_dense_4_bias: :
'read_8_disablecopyonread_dense_5_kernel:	 ј4
%read_9_disablecopyonread_dense_5_bias:	јM
3read_10_disablecopyonread_conv2d_transpose_5_kernel:``?
1read_11_disablecopyonread_conv2d_transpose_5_bias:`M
3read_12_disablecopyonread_conv2d_transpose_6_kernel:@`?
1read_13_disablecopyonread_conv2d_transpose_6_bias:@M
3read_14_disablecopyonread_conv2d_transpose_7_kernel:@?
1read_15_disablecopyonread_conv2d_transpose_7_bias:-
#read_16_disablecopyonread_adam_iter:	 /
%read_17_disablecopyonread_adam_beta_1: /
%read_18_disablecopyonread_adam_beta_2: .
$read_19_disablecopyonread_adam_decay: 6
,read_20_disablecopyonread_adam_learning_rate: +
!read_21_disablecopyonread_total_1: +
!read_22_disablecopyonread_count_1: )
read_23_disablecopyonread_total: )
read_24_disablecopyonread_count: J
0read_25_disablecopyonread_adam_conv2d_5_kernel_m: <
.read_26_disablecopyonread_adam_conv2d_5_bias_m: J
0read_27_disablecopyonread_adam_conv2d_6_kernel_m: @<
.read_28_disablecopyonread_adam_conv2d_6_bias_m:@J
0read_29_disablecopyonread_adam_conv2d_7_kernel_m:@`<
.read_30_disablecopyonread_adam_conv2d_7_bias_m:`B
/read_31_disablecopyonread_adam_dense_4_kernel_m:	ј ;
-read_32_disablecopyonread_adam_dense_4_bias_m: B
/read_33_disablecopyonread_adam_dense_5_kernel_m:	 ј<
-read_34_disablecopyonread_adam_dense_5_bias_m:	јT
:read_35_disablecopyonread_adam_conv2d_transpose_5_kernel_m:``F
8read_36_disablecopyonread_adam_conv2d_transpose_5_bias_m:`T
:read_37_disablecopyonread_adam_conv2d_transpose_6_kernel_m:@`F
8read_38_disablecopyonread_adam_conv2d_transpose_6_bias_m:@T
:read_39_disablecopyonread_adam_conv2d_transpose_7_kernel_m:@F
8read_40_disablecopyonread_adam_conv2d_transpose_7_bias_m:J
0read_41_disablecopyonread_adam_conv2d_5_kernel_v: <
.read_42_disablecopyonread_adam_conv2d_5_bias_v: J
0read_43_disablecopyonread_adam_conv2d_6_kernel_v: @<
.read_44_disablecopyonread_adam_conv2d_6_bias_v:@J
0read_45_disablecopyonread_adam_conv2d_7_kernel_v:@`<
.read_46_disablecopyonread_adam_conv2d_7_bias_v:`B
/read_47_disablecopyonread_adam_dense_4_kernel_v:	ј ;
-read_48_disablecopyonread_adam_dense_4_bias_v: B
/read_49_disablecopyonread_adam_dense_5_kernel_v:	 ј<
-read_50_disablecopyonread_adam_dense_5_bias_v:	јT
:read_51_disablecopyonread_adam_conv2d_transpose_5_kernel_v:``F
8read_52_disablecopyonread_adam_conv2d_transpose_5_bias_v:`T
:read_53_disablecopyonread_adam_conv2d_transpose_6_kernel_v:@`F
8read_54_disablecopyonread_adam_conv2d_transpose_6_bias_v:@T
:read_55_disablecopyonread_adam_conv2d_transpose_7_kernel_v:@F
8read_56_disablecopyonread_adam_conv2d_transpose_7_bias_v:
savev2_const
identity_115ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 ™
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_5_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_5_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_6_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: @z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_6_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_7_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_7_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:`{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 ®
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_4_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ј *
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ј f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	ј y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 °
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
 ®
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_5_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ј*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 јf
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	 јy
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_5_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ј*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:јb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:јИ
Read_10/DisableCopyOnReadDisableCopyOnRead3read_10_disablecopyonread_conv2d_transpose_5_kernel"/device:CPU:0*
_output_shapes
 љ
Read_10/ReadVariableOpReadVariableOp3read_10_disablecopyonread_conv2d_transpose_5_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:``*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:``m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:``Ж
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_conv2d_transpose_5_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_conv2d_transpose_5_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:`И
Read_12/DisableCopyOnReadDisableCopyOnRead3read_12_disablecopyonread_conv2d_transpose_6_kernel"/device:CPU:0*
_output_shapes
 љ
Read_12/ReadVariableOpReadVariableOp3read_12_disablecopyonread_conv2d_transpose_6_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`Ж
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_conv2d_transpose_6_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_conv2d_transpose_6_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_14/DisableCopyOnReadDisableCopyOnRead3read_14_disablecopyonread_conv2d_transpose_7_kernel"/device:CPU:0*
_output_shapes
 љ
Read_14/ReadVariableOpReadVariableOp3read_14_disablecopyonread_conv2d_transpose_7_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Ж
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_conv2d_transpose_7_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_conv2d_transpose_7_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
 Э
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
 Я
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
 Я
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
 Ю
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
: Б
Read_20/DisableCopyOnReadDisableCopyOnRead,read_20_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 ¶
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
 Ы
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
 Ы
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
 Щ
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
 Щ
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
: Е
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_conv2d_5_kernel_m"/device:CPU:0*
_output_shapes
 Ї
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_conv2d_5_kernel_m^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_conv2d_5_bias_m"/device:CPU:0*
_output_shapes
 ђ
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_conv2d_5_bias_m^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_conv2d_6_kernel_m"/device:CPU:0*
_output_shapes
 Ї
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_conv2d_6_kernel_m^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_conv2d_6_bias_m"/device:CPU:0*
_output_shapes
 ђ
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_conv2d_6_bias_m^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_conv2d_7_kernel_m"/device:CPU:0*
_output_shapes
 Ї
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_conv2d_7_kernel_m^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`Г
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_conv2d_7_bias_m"/device:CPU:0*
_output_shapes
 ђ
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_conv2d_7_bias_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:`Д
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_dense_4_kernel_m"/device:CPU:0*
_output_shapes
 ≤
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_dense_4_kernel_m^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ј *
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ј f
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	ј В
Read_32/DisableCopyOnReadDisableCopyOnRead-read_32_disablecopyonread_adam_dense_4_bias_m"/device:CPU:0*
_output_shapes
 Ђ
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
: Д
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_adam_dense_5_kernel_m"/device:CPU:0*
_output_shapes
 ≤
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_adam_dense_5_kernel_m^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ј*
dtype0p
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 јf
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	 јВ
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_dense_5_bias_m"/device:CPU:0*
_output_shapes
 ђ
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_dense_5_bias_m^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ј*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:јb
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:јП
Read_35/DisableCopyOnReadDisableCopyOnRead:read_35_disablecopyonread_adam_conv2d_transpose_5_kernel_m"/device:CPU:0*
_output_shapes
 ƒ
Read_35/ReadVariableOpReadVariableOp:read_35_disablecopyonread_adam_conv2d_transpose_5_kernel_m^Read_35/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:``*
dtype0w
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:``m
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*&
_output_shapes
:``Н
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_adam_conv2d_transpose_5_bias_m"/device:CPU:0*
_output_shapes
 ґ
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_adam_conv2d_transpose_5_bias_m^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:`П
Read_37/DisableCopyOnReadDisableCopyOnRead:read_37_disablecopyonread_adam_conv2d_transpose_6_kernel_m"/device:CPU:0*
_output_shapes
 ƒ
Read_37/ReadVariableOpReadVariableOp:read_37_disablecopyonread_adam_conv2d_transpose_6_kernel_m^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`Н
Read_38/DisableCopyOnReadDisableCopyOnRead8read_38_disablecopyonread_adam_conv2d_transpose_6_bias_m"/device:CPU:0*
_output_shapes
 ґ
Read_38/ReadVariableOpReadVariableOp8read_38_disablecopyonread_adam_conv2d_transpose_6_bias_m^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_39/DisableCopyOnReadDisableCopyOnRead:read_39_disablecopyonread_adam_conv2d_transpose_7_kernel_m"/device:CPU:0*
_output_shapes
 ƒ
Read_39/ReadVariableOpReadVariableOp:read_39_disablecopyonread_adam_conv2d_transpose_7_kernel_m^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Н
Read_40/DisableCopyOnReadDisableCopyOnRead8read_40_disablecopyonread_adam_conv2d_transpose_7_bias_m"/device:CPU:0*
_output_shapes
 ґ
Read_40/ReadVariableOpReadVariableOp8read_40_disablecopyonread_adam_conv2d_transpose_7_bias_m^Read_40/DisableCopyOnRead"/device:CPU:0*
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
:Е
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_conv2d_5_kernel_v"/device:CPU:0*
_output_shapes
 Ї
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_conv2d_5_kernel_v^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_conv2d_5_bias_v"/device:CPU:0*
_output_shapes
 ђ
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_conv2d_5_bias_v^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_conv2d_6_kernel_v"/device:CPU:0*
_output_shapes
 Ї
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_conv2d_6_kernel_v^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_conv2d_6_bias_v"/device:CPU:0*
_output_shapes
 ђ
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_conv2d_6_bias_v^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_conv2d_7_kernel_v"/device:CPU:0*
_output_shapes
 Ї
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_conv2d_7_kernel_v^Read_45/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0w
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`m
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`Г
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_conv2d_7_bias_v"/device:CPU:0*
_output_shapes
 ђ
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_conv2d_7_bias_v^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:`Д
Read_47/DisableCopyOnReadDisableCopyOnRead/read_47_disablecopyonread_adam_dense_4_kernel_v"/device:CPU:0*
_output_shapes
 ≤
Read_47/ReadVariableOpReadVariableOp/read_47_disablecopyonread_adam_dense_4_kernel_v^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ј *
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ј f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	ј В
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_dense_4_bias_v"/device:CPU:0*
_output_shapes
 Ђ
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
: Д
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_dense_5_kernel_v"/device:CPU:0*
_output_shapes
 ≤
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_dense_5_kernel_v^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ј*
dtype0p
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 јf
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:	 јВ
Read_50/DisableCopyOnReadDisableCopyOnRead-read_50_disablecopyonread_adam_dense_5_bias_v"/device:CPU:0*
_output_shapes
 ђ
Read_50/ReadVariableOpReadVariableOp-read_50_disablecopyonread_adam_dense_5_bias_v^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ј*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:јd
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:јП
Read_51/DisableCopyOnReadDisableCopyOnRead:read_51_disablecopyonread_adam_conv2d_transpose_5_kernel_v"/device:CPU:0*
_output_shapes
 ƒ
Read_51/ReadVariableOpReadVariableOp:read_51_disablecopyonread_adam_conv2d_transpose_5_kernel_v^Read_51/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:``*
dtype0x
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:``o
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*&
_output_shapes
:``Н
Read_52/DisableCopyOnReadDisableCopyOnRead8read_52_disablecopyonread_adam_conv2d_transpose_5_bias_v"/device:CPU:0*
_output_shapes
 ґ
Read_52/ReadVariableOpReadVariableOp8read_52_disablecopyonread_adam_conv2d_transpose_5_bias_v^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:`П
Read_53/DisableCopyOnReadDisableCopyOnRead:read_53_disablecopyonread_adam_conv2d_transpose_6_kernel_v"/device:CPU:0*
_output_shapes
 ƒ
Read_53/ReadVariableOpReadVariableOp:read_53_disablecopyonread_adam_conv2d_transpose_6_kernel_v^Read_53/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0x
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`o
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`Н
Read_54/DisableCopyOnReadDisableCopyOnRead8read_54_disablecopyonread_adam_conv2d_transpose_6_bias_v"/device:CPU:0*
_output_shapes
 ґ
Read_54/ReadVariableOpReadVariableOp8read_54_disablecopyonread_adam_conv2d_transpose_6_bias_v^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_55/DisableCopyOnReadDisableCopyOnRead:read_55_disablecopyonread_adam_conv2d_transpose_7_kernel_v"/device:CPU:0*
_output_shapes
 ƒ
Read_55/ReadVariableOpReadVariableOp:read_55_disablecopyonread_adam_conv2d_transpose_7_kernel_v^Read_55/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Н
Read_56/DisableCopyOnReadDisableCopyOnRead8read_56_disablecopyonread_adam_conv2d_transpose_7_bias_v"/device:CPU:0*
_output_shapes
 ґ
Read_56/ReadVariableOpReadVariableOp8read_56_disablecopyonread_adam_conv2d_transpose_7_bias_v^Read_56/DisableCopyOnRead"/device:CPU:0*
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
:ў
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*В
valueшBх:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*З
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
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
: Р
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_115Identity_115:output:0*Й
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
ё

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273682

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€`i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
Ґ
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_2273698

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€јY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
ѕ
c
G__inference_resizing_2_layer_call_and_return_conditional_losses_2273959

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   Ъ
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€@А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ
≈
%__inference_signature_wrapper_2272728
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
	unknown_7:	 ј
	unknown_8:	ј#
	unknown_9:``

unknown_10:`$

unknown_11:@`

unknown_12:@$

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallь
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
:€€€€€€€€€@А*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_2271420x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
≠
е
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272607
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
	unknown_7:	 ј
	unknown_8:	ј#
	unknown_9:``

unknown_10:`$

unknown_11:@`

unknown_12:@$

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallљ
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
:€€€€€€€€€@А: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *i
fdRb
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272571x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
ё

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271567

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€`i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
†

ч
D__inference_dense_5_layer_call_and_return_conditional_losses_2273749

inputs1
matmul_readvariableop_resource:	 ј.
biasadd_readvariableop_resource:	ј
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јO
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јa
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€јw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
У
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
Я
*__inference_conv2d_7_layer_call_fn_2273639

inputs!
unknown:@`
	unknown_0:`
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ї
M
1__inference_max_pooling2d_5_layer_call_fn_2273541

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґ
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2273948

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
…
©
4__inference_conv2d_transpose_7_layer_call_fn_2273897

inputs!
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ё
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€``
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
х(
р
D__inference_Decoder_layer_call_and_return_conditional_losses_2272236

inputs"
dense_5_2272210:	 ј
dense_5_2272212:	ј4
conv2d_transpose_5_2272216:``(
conv2d_transpose_5_2272218:`4
conv2d_transpose_6_2272222:@`(
conv2d_transpose_6_2272224:@4
conv2d_transpose_7_2272228:@(
conv2d_transpose_7_2272230:
identityИҐ*conv2d_transpose_5/StatefulPartitionedCallҐ*conv2d_transpose_6/StatefulPartitionedCallҐ*conv2d_transpose_7/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallр
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_2272210dense_5_2272212*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126е
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146њ
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_5_2272216conv2d_transpose_5_2272218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954О
up_sampling2d_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977„
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_transpose_6_2272222conv2d_transpose_6_2272224*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018О
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041„
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_2272228conv2d_transpose_7_2272230*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082О
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105и
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Ап
NoOpNoOp+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
щ
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273687

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€`c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
У
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2273660

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
О8
•
D__inference_Encoder_layer_call_and_return_conditional_losses_2271764

inputs*
conv2d_5_2271727: 
conv2d_5_2271729: *
conv2d_6_2271734: @
conv2d_6_2271736:@*
conv2d_7_2271741:@`
conv2d_7_2271743:`"
dense_4_2271749:	ј 
dense_4_2271751: 
identity

identity_1ИҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallы
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_2271727conv2d_5_2271729*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ @ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484т
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426б
dropout/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271617Х
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_6_2271734conv2d_6_2271736*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516т
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438е
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271629Ч
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_7_2271741conv2d_7_2271743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548т
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450е
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271641Ў
flatten_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575Л
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_2271749dense_4_2271751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588»
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
 *-
config_proto

CPU

GPU 2J 8В *9
f4R2
0__inference_dense_4_activity_regularizer_2271469З
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ђ
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: —
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
„V
а
D__inference_Encoder_layer_call_and_return_conditional_losses_2273231

inputsA
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@`6
(conv2d_7_biasadd_readvariableop_resource:`9
&dense_4_matmul_readvariableop_resource:	ј 5
'dense_4_biasadd_readvariableop_resource: 
identity

identity_1ИҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpО
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ђ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ h
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ ™
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Elu:activations:0*/
_output_shapes
:€€€€€€€€€  *
ksize
*
paddingSAME*
strides
Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?Ц
dropout/dropout/MulMul max_pooling2d_5/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€  s
dropout/dropout/ShapeShape max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
::нѕ∞
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€  *
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>∆
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€  \
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ї
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€  О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0∆
conv2d_6/Conv2DConv2D!dropout/dropout/SelectV2:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@h
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@™
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Elu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?Ъ
dropout_1/dropout/MulMul max_pooling2d_6/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@u
dropout_1/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
::нѕЅ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype0*

seed**
seed2e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ћ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@О
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0»
conv2d_7/Conv2DConv2D#dropout_1/dropout/SelectV2:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Д
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ш
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`h
conv2d_7/EluEluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`™
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Elu:activations:0*/
_output_shapes
:€€€€€€€€€`*
ksize
*
paddingSAME*
strides
\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?Ъ
dropout_2/dropout/MulMul max_pooling2d_7/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€`u
dropout_2/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
::нѕЅ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
dtype0*

seed**
seed2e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ћ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€`^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€``
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   О
flatten_2/ReshapeReshape#dropout_2/dropout/SelectV2:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€јЕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0Н
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
dense_4/EluEludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ s
dense_4/ActivityRegularizer/AbsAbsdense_4/Elu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ r
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/Abs:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: x
!dense_4/ActivityRegularizer/ShapeShapedense_4/Elu:activations:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ъ
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: h
IdentityIdentitydense_4/Elu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: –
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
З©
‘
"__inference__wrapped_model_2271420
input_9m
Sae_conv_prep_flatten_stft_augmented_encoder_conv2d_5_conv2d_readvariableop_resource: b
Tae_conv_prep_flatten_stft_augmented_encoder_conv2d_5_biasadd_readvariableop_resource: m
Sae_conv_prep_flatten_stft_augmented_encoder_conv2d_6_conv2d_readvariableop_resource: @b
Tae_conv_prep_flatten_stft_augmented_encoder_conv2d_6_biasadd_readvariableop_resource:@m
Sae_conv_prep_flatten_stft_augmented_encoder_conv2d_7_conv2d_readvariableop_resource:@`b
Tae_conv_prep_flatten_stft_augmented_encoder_conv2d_7_biasadd_readvariableop_resource:`e
Rae_conv_prep_flatten_stft_augmented_encoder_dense_4_matmul_readvariableop_resource:	ј a
Sae_conv_prep_flatten_stft_augmented_encoder_dense_4_biasadd_readvariableop_resource: e
Rae_conv_prep_flatten_stft_augmented_decoder_dense_5_matmul_readvariableop_resource:	 јb
Sae_conv_prep_flatten_stft_augmented_decoder_dense_5_biasadd_readvariableop_resource:	јБ
gae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:``l
^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_5_biasadd_readvariableop_resource:`Б
gae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@`l
^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_6_biasadd_readvariableop_resource:@Б
gae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@l
^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_7_biasadd_readvariableop_resource:
identityИҐUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpҐ^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpҐUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpҐ^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpҐUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpҐ^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpҐJAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOpҐIAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOpҐKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOpҐJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOpҐKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOpҐJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOpҐKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOpҐJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOpҐJAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOpҐIAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOpж
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOpSae_conv_prep_flatten_stft_augmented_encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Д
;AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2DConv2Dinput_9RAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
paddingSAME*
strides
№
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_stft_augmented_encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ь
<AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAddBiasAddDAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D:output:0SAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ ј
8AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/EluEluEAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ В
CAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_5/MaxPoolMaxPoolFAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Elu:activations:0*/
_output_shapes
:€€€€€€€€€  *
ksize
*
paddingSAME*
strides
–
<AE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout/IdentityIdentityLAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€  ж
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOpSae_conv_prep_flatten_stft_augmented_encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¬
;AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2DConv2DEAE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout/Identity:output:0RAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
№
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_stft_augmented_encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ь
<AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAddBiasAddDAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D:output:0SAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@ј
8AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/EluEluEAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@В
CAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_6/MaxPoolMaxPoolFAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Elu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
“
>AE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout_1/IdentityIdentityLAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@ж
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOpSae_conv_prep_flatten_stft_augmented_encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ƒ
;AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2DConv2DGAE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout_1/Identity:output:0RAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
№
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_stft_augmented_encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ь
<AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAddBiasAddDAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D:output:0SAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`ј
8AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/EluEluEAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`В
CAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_7/MaxPoolMaxPoolFAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Elu:activations:0*/
_output_shapes
:€€€€€€€€€`*
ksize
*
paddingSAME*
strides
“
>AE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout_2/IdentityIdentityLAE_Conv_prep_flatten_STFT_Augmented/Encoder/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€`М
;AE_Conv_prep_flatten_STFT_Augmented/Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   К
=AE_Conv_prep_flatten_STFT_Augmented/Encoder/flatten_2/ReshapeReshapeGAE_Conv_prep_flatten_STFT_Augmented/Encoder/dropout_2/Identity:output:0DAE_Conv_prep_flatten_STFT_Augmented/Encoder/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€јЁ
IAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOpReadVariableOpRae_conv_prep_flatten_stft_augmented_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0С
:AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMulMatMulFAE_Conv_prep_flatten_STFT_Augmented/Encoder/flatten_2/Reshape:output:0QAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Џ
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpSae_conv_prep_flatten_stft_augmented_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
;AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAddBiasAddDAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul:product:0RAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ґ
7AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/EluEluDAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ћ
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/AbsAbsEAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/Elu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
MAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/SumSumOAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/Abs:y:0VAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: Т
MAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:°
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/mulMulVAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/mul/x:output:0TAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: –
MAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/ShapeShapeEAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/Elu:activations:0*
T0*
_output_shapes
::нѕ•
[AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: І
]AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:І
]AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
UAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSliceVAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/Shape:output:0dAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0fAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0fAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskд
LAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/CastCast^AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ю
OAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/truedivRealDivOAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/mul:z:0PAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ё
IAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOpReadVariableOpRae_conv_prep_flatten_stft_augmented_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0С
:AE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMulMatMulEAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/Elu:activations:0QAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јџ
JAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOpSae_conv_prep_flatten_stft_augmented_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0У
;AE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAddBiasAddDAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul:product:0RAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јЈ
7AE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/EluEluDAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јЊ
;AE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/ShapeShapeEAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/Elu:activations:0*
T0*
_output_shapes
::нѕУ
IAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Х
KAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Х
KAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
CAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_sliceStridedSliceDAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Shape:output:0RAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stack:output:0TAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stack_1:output:0TAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЗ
EAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
EAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :З
EAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`Ј
CAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shapePackLAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/strided_slice:output:0NAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/1:output:0NAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/2:output:0NAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ч
=AE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/ReshapeReshapeEAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/Elu:activations:0LAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`»
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/ShapeShapeFAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::нѕЬ
RAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
LAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_sliceStridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/Shape:output:0[AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stack_1:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`ƒ
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stackPackUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/1:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/2:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: †
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:†
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
NAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1StridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stack:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpgae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0«
OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInputMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/stack:output:0fAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0FAE_Conv_prep_flatten_STFT_Augmented/Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
р
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ƒ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAddBiasAddXAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`‘
BAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/EluEluOAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Т
AAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      Ф
CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Е
?AE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/mulMulJAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/Const:output:0LAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:№
XAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighborPAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/Elu:activations:0CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€`*
half_pixel_centers(л
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/ShapeShapeiAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕЬ
RAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
LAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_sliceStridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/Shape:output:0[AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stack_1:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@ƒ
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stackPackUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/1:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/2:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: †
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:†
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
NAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1StridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpgae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0к
OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/stack:output:0fAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0iAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
р
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ƒ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAddBiasAddXAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@‘
BAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/EluEluOAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Т
AAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      Ф
CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Е
?AE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/mulMulJAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/Const:output:0LAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:№
XAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborPAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/Elu:activations:0CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
half_pixel_centers(л
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/ShapeShapeiAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕЬ
RAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
LAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_sliceStridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/Shape:output:0[AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stack_1:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B : И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@И
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ƒ
DAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stackPackUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/1:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/2:output:0OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:Ю
TAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: †
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:†
VAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
NAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1StridedSliceMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0_AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpgae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0к
OAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInputMAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/stack:output:0fAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0iAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
paddingSAME*
strides
р
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp^ae_conv_prep_flatten_stft_augmented_decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ƒ
FAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAddBiasAddXAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose:output:0]AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @‘
BAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/EluEluOAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @Т
AAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   Ф
CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Е
?AE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/mulMulJAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/Const:output:0LAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:Ё
XAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborPAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/Elu:activations:0CAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(У
BAE_Conv_prep_flatten_STFT_Augmented/Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   л
LAE_Conv_prep_flatten_STFT_Augmented/Decoder/resizing_2/resize/ResizeBilinearResizeBilineariAE_Conv_prep_flatten_STFT_Augmented/Decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0KAE_Conv_prep_flatten_STFT_Augmented/Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(µ
IdentityIdentity]AE_Conv_prep_flatten_STFT_Augmented/Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аф

NoOpNoOpV^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp_^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpV^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp_^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpV^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp_^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpK^AE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOpJ^AE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOpL^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOpK^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOpL^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOpK^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOpL^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOpK^AE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOpK^AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOpJ^AE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2Ѓ
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2ј
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2Ѓ
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2ј
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2Ѓ
UAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpUAE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2ј
^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp^AE_Conv_prep_flatten_STFT_Augmented/Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2Ш
JAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOpJAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/BiasAdd/ReadVariableOp2Ц
IAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOpIAE_Conv_prep_flatten_STFT_Augmented/Decoder/dense_5/MatMul/ReadVariableOp2Ъ
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/BiasAdd/ReadVariableOp2Ш
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOpJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_5/Conv2D/ReadVariableOp2Ъ
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/BiasAdd/ReadVariableOp2Ш
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOpJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_6/Conv2D/ReadVariableOp2Ъ
KAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/BiasAdd/ReadVariableOp2Ш
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOpJAE_Conv_prep_flatten_STFT_Augmented/Encoder/conv2d_7/Conv2D/ReadVariableOp2Ш
JAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOpJAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/BiasAdd/ReadVariableOp2Ц
IAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOpIAE_Conv_prep_flatten_STFT_Augmented/Encoder/dense_4/MatMul/ReadVariableOp:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
С8
¶
D__inference_Encoder_layer_call_and_return_conditional_losses_2271659
input_7*
conv2d_5_2271607: 
conv2d_5_2271609: *
conv2d_6_2271619: @
conv2d_6_2271621:@*
conv2d_7_2271631:@`
conv2d_7_2271633:`"
dense_4_2271644:	ј 
dense_4_2271646: 
identity

identity_1ИҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallь
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_5_2271607conv2d_5_2271609*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ @ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484т
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2271426б
dropout/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271617Х
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_6_2271619conv2d_6_2271621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516т
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438е
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271629Ч
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_7_2271631conv2d_7_2271633*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2271548т
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450е
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271641Ў
flatten_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575Л
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_2271644dense_4_2271646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588»
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
 *-
config_proto

CPU

GPU 2J 8В *9
f4R2
0__inference_dense_4_activity_regularizer_2271469З
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ђ
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: —
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_7
п	
“
)__inference_Encoder_layer_call_fn_2271784
input_7!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_7
й
H
,__inference_resizing_2_layer_call_fn_2273953

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈
Ч
)__inference_dense_4_layer_call_fn_2273707

inputs
unknown:	ј 
	unknown_0: 
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
ш(
с
D__inference_Decoder_layer_call_and_return_conditional_losses_2272204
input_8"
dense_5_2272178:	 ј
dense_5_2272180:	ј4
conv2d_transpose_5_2272184:``(
conv2d_transpose_5_2272186:`4
conv2d_transpose_6_2272190:@`(
conv2d_transpose_6_2272192:@4
conv2d_transpose_7_2272196:@(
conv2d_transpose_7_2272198:
identityИҐ*conv2d_transpose_5/StatefulPartitionedCallҐ*conv2d_transpose_6/StatefulPartitionedCallҐ*conv2d_transpose_7/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallс
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_5_2272178dense_5_2272180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126е
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146њ
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_5_2272184conv2d_transpose_5_2272186*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954О
up_sampling2d_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977„
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_transpose_6_2272190conv2d_transpose_6_2272192*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018О
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041„
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_2272196conv2d_transpose_7_2272198*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082О
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105и
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Ап
NoOpNoOp+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€ 
!
_user_specified_name	input_8
Ї
M
1__inference_up_sampling2d_5_layer_call_fn_2273816

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≠
е
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272529
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
	unknown_7:	 ј
	unknown_8:	ј#
	unknown_9:``

unknown_10:`$

unknown_11:@`

unknown_12:@$

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallљ
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
:€€€€€€€€€@А: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *i
fdRb
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272493x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
Ґ
h
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
b
D__inference_dropout_layer_call_and_return_conditional_losses_2271617

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Ґ
h
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2273828

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
ю
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
≥
G
+__inference_flatten_2_layer_call_fn_2273692

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
к	
“
)__inference_Decoder_layer_call_fn_2273305

inputs
unknown:	 ј
	unknown_0:	ј#
	unknown_1:``
	unknown_2:`#
	unknown_3:@`
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272236x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
№

c
D__inference_dropout_layer_call_and_return_conditional_losses_2273568

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€  *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€  i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
к	
“
)__inference_Decoder_layer_call_fn_2273326

inputs
unknown:	 ј
	unknown_0:	ј#
	unknown_1:``
	unknown_2:`#
	unknown_3:@`
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272286x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
÷»
Ю
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2273113

inputsI
/encoder_conv2d_5_conv2d_readvariableop_resource: >
0encoder_conv2d_5_biasadd_readvariableop_resource: I
/encoder_conv2d_6_conv2d_readvariableop_resource: @>
0encoder_conv2d_6_biasadd_readvariableop_resource:@I
/encoder_conv2d_7_conv2d_readvariableop_resource:@`>
0encoder_conv2d_7_biasadd_readvariableop_resource:`A
.encoder_dense_4_matmul_readvariableop_resource:	ј =
/encoder_dense_4_biasadd_readvariableop_resource: A
.decoder_dense_5_matmul_readvariableop_resource:	 ј>
/decoder_dense_5_biasadd_readvariableop_resource:	ј]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:``H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:`]
Cdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@`H
:decoder_conv2d_transpose_6_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@H
:decoder_conv2d_transpose_7_biasadd_readvariableop_resource:
identity

identity_1ИҐ1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpҐ1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpҐ1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpҐ&Decoder/dense_5/BiasAdd/ReadVariableOpҐ%Decoder/dense_5/MatMul/ReadVariableOpҐ'Encoder/conv2d_5/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_5/Conv2D/ReadVariableOpҐ'Encoder/conv2d_6/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_6/Conv2D/ReadVariableOpҐ'Encoder/conv2d_7/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_7/Conv2D/ReadVariableOpҐ&Encoder/dense_4/BiasAdd/ReadVariableOpҐ%Encoder/dense_4/MatMul/ReadVariableOpЮ
&Encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ї
Encoder/conv2d_5/Conv2DConv2Dinputs.Encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
paddingSAME*
strides
Ф
'Encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0∞
Encoder/conv2d_5/BiasAddBiasAdd Encoder/conv2d_5/Conv2D:output:0/Encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ x
Encoder/conv2d_5/EluElu!Encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ Ї
Encoder/max_pooling2d_5/MaxPoolMaxPool"Encoder/conv2d_5/Elu:activations:0*/
_output_shapes
:€€€€€€€€€  *
ksize
*
paddingSAME*
strides
И
Encoder/dropout/IdentityIdentity(Encoder/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€  Ю
&Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0÷
Encoder/conv2d_6/Conv2DConv2D!Encoder/dropout/Identity:output:0.Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Ф
'Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0∞
Encoder/conv2d_6/BiasAddBiasAdd Encoder/conv2d_6/Conv2D:output:0/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@x
Encoder/conv2d_6/EluElu!Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ї
Encoder/max_pooling2d_6/MaxPoolMaxPool"Encoder/conv2d_6/Elu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
К
Encoder/dropout_1/IdentityIdentity(Encoder/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ю
&Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0Ў
Encoder/conv2d_7/Conv2DConv2D#Encoder/dropout_1/Identity:output:0.Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Ф
'Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0∞
Encoder/conv2d_7/BiasAddBiasAdd Encoder/conv2d_7/Conv2D:output:0/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`x
Encoder/conv2d_7/EluElu!Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Ї
Encoder/max_pooling2d_7/MaxPoolMaxPool"Encoder/conv2d_7/Elu:activations:0*/
_output_shapes
:€€€€€€€€€`*
ksize
*
paddingSAME*
strides
К
Encoder/dropout_2/IdentityIdentity(Encoder/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€`h
Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   Ю
Encoder/flatten_2/ReshapeReshape#Encoder/dropout_2/Identity:output:0 Encoder/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€јХ
%Encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0•
Encoder/dense_4/MatMulMatMul"Encoder/flatten_2/Reshape:output:0-Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Т
&Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
Encoder/dense_4/BiasAddBiasAdd Encoder/dense_4/MatMul:product:0.Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ n
Encoder/dense_4/EluElu Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
'Encoder/dense_4/ActivityRegularizer/AbsAbs!Encoder/dense_4/Elu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ z
)Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ∞
'Encoder/dense_4/ActivityRegularizer/SumSum+Encoder/dense_4/ActivityRegularizer/Abs:y:02Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:µ
'Encoder/dense_4/ActivityRegularizer/mulMul2Encoder/dense_4/ActivityRegularizer/mul/x:output:00Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: И
)Encoder/dense_4/ActivityRegularizer/ShapeShape!Encoder/dense_4/Elu:activations:0*
T0*
_output_shapes
::нѕБ
7Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
1Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_4/ActivityRegularizer/Shape:output:0@Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЬ
(Encoder/dense_4/ActivityRegularizer/CastCast:Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ≤
+Encoder/dense_4/ActivityRegularizer/truedivRealDiv+Encoder/dense_4/ActivityRegularizer/mul:z:0,Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Х
%Decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0•
Decoder/dense_5/MatMulMatMul!Encoder/dense_4/Elu:activations:0-Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јУ
&Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0І
Decoder/dense_5/BiasAddBiasAdd Decoder/dense_5/MatMul:product:0.Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јo
Decoder/dense_5/EluElu Decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јv
Decoder/reshape_2/ShapeShape!Decoder/dense_5/Elu:activations:0*
T0*
_output_shapes
::нѕo
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
valueB:Ђ
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
value	B :c
!Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`Г
Decoder/reshape_2/Reshape/shapePack(Decoder/reshape_2/strided_slice:output:0*Decoder/reshape_2/Reshape/shape/1:output:0*Decoder/reshape_2/Reshape/shape/2:output:0*Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
Decoder/reshape_2/ReshapeReshape!Decoder/dense_5/Elu:activations:0(Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`А
 Decoder/conv2d_transpose_5/ShapeShape"Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::нѕx
.Decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(Decoder/conv2d_transpose_5/strided_sliceStridedSlice)Decoder/conv2d_transpose_5/Shape:output:07Decoder/conv2d_transpose_5/strided_slice/stack:output:09Decoder/conv2d_transpose_5/strided_slice/stack_1:output:09Decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`Р
 Decoder/conv2d_transpose_5/stackPack1Decoder/conv2d_transpose_5/strided_slice:output:0+Decoder/conv2d_transpose_5/stack/1:output:0+Decoder/conv2d_transpose_5/stack/2:output:0+Decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
*Decoder/conv2d_transpose_5/strided_slice_1StridedSlice)Decoder/conv2d_transpose_5/stack:output:09Decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0Ј
+Decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_5/stack:output:0BDecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ў
"Decoder/conv2d_transpose_5/BiasAddBiasAdd4Decoder/conv2d_transpose_5/conv2d_transpose:output:09Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`М
Decoder/conv2d_transpose_5/EluElu+Decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`n
Decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_5/mulMul&Decoder/up_sampling2d_5/Const:output:0(Decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:р
4Decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_5/Elu:activations:0Decoder/up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€`*
half_pixel_centers(£
 Decoder/conv2d_transpose_6/ShapeShapeEDecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕx
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
valueB:Ў
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
value	B :d
"Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Р
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
valueB:а
*Decoder/conv2d_transpose_6/strided_slice_1StridedSlice)Decoder/conv2d_transpose_6/stack:output:09Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0Џ
+Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_6/stack:output:0BDecoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
"Decoder/conv2d_transpose_6/BiasAddBiasAdd4Decoder/conv2d_transpose_6/conv2d_transpose:output:09Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@М
Decoder/conv2d_transpose_6/EluElu+Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@n
Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_6/mulMul&Decoder/up_sampling2d_6/Const:output:0(Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:р
4Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_6/Elu:activations:0Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
half_pixel_centers(£
 Decoder/conv2d_transpose_7/ShapeShapeEDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕx
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
valueB:Ў
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
value	B : d
"Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Р
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
valueB:а
*Decoder/conv2d_transpose_7/strided_slice_1StridedSlice)Decoder/conv2d_transpose_7/stack:output:09Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Џ
+Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_7/stack:output:0BDecoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
"Decoder/conv2d_transpose_7/BiasAddBiasAdd4Decoder/conv2d_transpose_7/conv2d_transpose:output:09Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @М
Decoder/conv2d_transpose_7/EluElu+Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @n
Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   p
Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_7/mulMul&Decoder/up_sampling2d_7/Const:output:0(Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:с
4Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_7/Elu:activations:0Decoder/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(o
Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   €
(Decoder/resizing_2/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(С
IdentityIdentity9Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аo

Identity_1Identity/Encoder/dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: і
NoOpNoOp2^Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp'^Decoder/dense_5/BiasAdd/ReadVariableOp&^Decoder/dense_5/MatMul/ReadVariableOp(^Encoder/conv2d_5/BiasAdd/ReadVariableOp'^Encoder/conv2d_5/Conv2D/ReadVariableOp(^Encoder/conv2d_6/BiasAdd/ReadVariableOp'^Encoder/conv2d_6/Conv2D/ReadVariableOp(^Encoder/conv2d_7/BiasAdd/ReadVariableOp'^Encoder/conv2d_7/Conv2D/ReadVariableOp'^Encoder/dense_4/BiasAdd/ReadVariableOp&^Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2f
1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_5/BiasAdd/ReadVariableOp&Decoder/dense_5/BiasAdd/ReadVariableOp2N
%Decoder/dense_5/MatMul/ReadVariableOp%Decoder/dense_5/MatMul/ReadVariableOp2R
'Encoder/conv2d_5/BiasAdd/ReadVariableOp'Encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_5/Conv2D/ReadVariableOp&Encoder/conv2d_5/Conv2D/ReadVariableOp2R
'Encoder/conv2d_6/BiasAdd/ReadVariableOp'Encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_6/Conv2D/ReadVariableOp&Encoder/conv2d_6/Conv2D/ReadVariableOp2R
'Encoder/conv2d_7/BiasAdd/ReadVariableOp'Encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_7/Conv2D/ReadVariableOp&Encoder/conv2d_7/Conv2D/ReadVariableOp2P
&Encoder/dense_4/BiasAdd/ReadVariableOp&Encoder/dense_4/BiasAdd/ReadVariableOp2N
%Encoder/dense_4/MatMul/ReadVariableOp%Encoder/dense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
Ї
M
1__inference_max_pooling2d_7_layer_call_fn_2273655

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2271450Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґ
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2273888

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ї
M
1__inference_up_sampling2d_6_layer_call_fn_2273876

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Г
ю
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2273536

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
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
:€€€€€€€€€ @ V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ @ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€@А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
™
д
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272766

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
	unknown_7:	 ј
	unknown_8:	ј#
	unknown_9:``

unknown_10:`$

unknown_11:@`

unknown_12:@$

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallЉ
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
:€€€€€€€€€@А: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *i
fdRb
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272493x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
Ё
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_2273768

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€``
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
пг
Ю
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272969

inputsI
/encoder_conv2d_5_conv2d_readvariableop_resource: >
0encoder_conv2d_5_biasadd_readvariableop_resource: I
/encoder_conv2d_6_conv2d_readvariableop_resource: @>
0encoder_conv2d_6_biasadd_readvariableop_resource:@I
/encoder_conv2d_7_conv2d_readvariableop_resource:@`>
0encoder_conv2d_7_biasadd_readvariableop_resource:`A
.encoder_dense_4_matmul_readvariableop_resource:	ј =
/encoder_dense_4_biasadd_readvariableop_resource: A
.decoder_dense_5_matmul_readvariableop_resource:	 ј>
/decoder_dense_5_biasadd_readvariableop_resource:	ј]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:``H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:`]
Cdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@`H
:decoder_conv2d_transpose_6_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@H
:decoder_conv2d_transpose_7_biasadd_readvariableop_resource:
identity

identity_1ИҐ1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpҐ1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpҐ1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpҐ:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpҐ&Decoder/dense_5/BiasAdd/ReadVariableOpҐ%Decoder/dense_5/MatMul/ReadVariableOpҐ'Encoder/conv2d_5/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_5/Conv2D/ReadVariableOpҐ'Encoder/conv2d_6/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_6/Conv2D/ReadVariableOpҐ'Encoder/conv2d_7/BiasAdd/ReadVariableOpҐ&Encoder/conv2d_7/Conv2D/ReadVariableOpҐ&Encoder/dense_4/BiasAdd/ReadVariableOpҐ%Encoder/dense_4/MatMul/ReadVariableOpЮ
&Encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ї
Encoder/conv2d_5/Conv2DConv2Dinputs.Encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
paddingSAME*
strides
Ф
'Encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0∞
Encoder/conv2d_5/BiasAddBiasAdd Encoder/conv2d_5/Conv2D:output:0/Encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ x
Encoder/conv2d_5/EluElu!Encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ Ї
Encoder/max_pooling2d_5/MaxPoolMaxPool"Encoder/conv2d_5/Elu:activations:0*/
_output_shapes
:€€€€€€€€€  *
ksize
*
paddingSAME*
strides
b
Encoder/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?Ѓ
Encoder/dropout/dropout/MulMul(Encoder/max_pooling2d_5/MaxPool:output:0&Encoder/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€  Г
Encoder/dropout/dropout/ShapeShape(Encoder/max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
::нѕј
4Encoder/dropout/dropout/random_uniform/RandomUniformRandomUniform&Encoder/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€  *
dtype0*

seed*k
&Encoder/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ё
$Encoder/dropout/dropout/GreaterEqualGreaterEqual=Encoder/dropout/dropout/random_uniform/RandomUniform:output:0/Encoder/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€  d
Encoder/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
 Encoder/dropout/dropout/SelectV2SelectV2(Encoder/dropout/dropout/GreaterEqual:z:0Encoder/dropout/dropout/Mul:z:0(Encoder/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€  Ю
&Encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ё
Encoder/conv2d_6/Conv2DConv2D)Encoder/dropout/dropout/SelectV2:output:0.Encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Ф
'Encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0∞
Encoder/conv2d_6/BiasAddBiasAdd Encoder/conv2d_6/Conv2D:output:0/Encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@x
Encoder/conv2d_6/EluElu!Encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ї
Encoder/max_pooling2d_6/MaxPoolMaxPool"Encoder/conv2d_6/Elu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
d
Encoder/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?≤
Encoder/dropout_1/dropout/MulMul(Encoder/max_pooling2d_6/MaxPool:output:0(Encoder/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Е
Encoder/dropout_1/dropout/ShapeShape(Encoder/max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
::нѕ—
6Encoder/dropout_1/dropout/random_uniform/RandomUniformRandomUniform(Encoder/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype0*

seed**
seed2m
(Encoder/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>д
&Encoder/dropout_1/dropout/GreaterEqualGreaterEqual?Encoder/dropout_1/dropout/random_uniform/RandomUniform:output:01Encoder/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@f
!Encoder/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    г
"Encoder/dropout_1/dropout/SelectV2SelectV2*Encoder/dropout_1/dropout/GreaterEqual:z:0!Encoder/dropout_1/dropout/Mul:z:0*Encoder/dropout_1/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ю
&Encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0а
Encoder/conv2d_7/Conv2DConv2D+Encoder/dropout_1/dropout/SelectV2:output:0.Encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Ф
'Encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0∞
Encoder/conv2d_7/BiasAddBiasAdd Encoder/conv2d_7/Conv2D:output:0/Encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`x
Encoder/conv2d_7/EluElu!Encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Ї
Encoder/max_pooling2d_7/MaxPoolMaxPool"Encoder/conv2d_7/Elu:activations:0*/
_output_shapes
:€€€€€€€€€`*
ksize
*
paddingSAME*
strides
d
Encoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?≤
Encoder/dropout_2/dropout/MulMul(Encoder/max_pooling2d_7/MaxPool:output:0(Encoder/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€`Е
Encoder/dropout_2/dropout/ShapeShape(Encoder/max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
::нѕ—
6Encoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform(Encoder/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
dtype0*

seed**
seed2m
(Encoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>д
&Encoder/dropout_2/dropout/GreaterEqualGreaterEqual?Encoder/dropout_2/dropout/random_uniform/RandomUniform:output:01Encoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€`f
!Encoder/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    г
"Encoder/dropout_2/dropout/SelectV2SelectV2*Encoder/dropout_2/dropout/GreaterEqual:z:0!Encoder/dropout_2/dropout/Mul:z:0*Encoder/dropout_2/dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€`h
Encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   ¶
Encoder/flatten_2/ReshapeReshape+Encoder/dropout_2/dropout/SelectV2:output:0 Encoder/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€јХ
%Encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0•
Encoder/dense_4/MatMulMatMul"Encoder/flatten_2/Reshape:output:0-Encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Т
&Encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
Encoder/dense_4/BiasAddBiasAdd Encoder/dense_4/MatMul:product:0.Encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ n
Encoder/dense_4/EluElu Encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
'Encoder/dense_4/ActivityRegularizer/AbsAbs!Encoder/dense_4/Elu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ z
)Encoder/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ∞
'Encoder/dense_4/ActivityRegularizer/SumSum+Encoder/dense_4/ActivityRegularizer/Abs:y:02Encoder/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:µ
'Encoder/dense_4/ActivityRegularizer/mulMul2Encoder/dense_4/ActivityRegularizer/mul/x:output:00Encoder/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: И
)Encoder/dense_4/ActivityRegularizer/ShapeShape!Encoder/dense_4/Elu:activations:0*
T0*
_output_shapes
::нѕБ
7Encoder/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9Encoder/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
1Encoder/dense_4/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_4/ActivityRegularizer/Shape:output:0@Encoder/dense_4/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЬ
(Encoder/dense_4/ActivityRegularizer/CastCast:Encoder/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ≤
+Encoder/dense_4/ActivityRegularizer/truedivRealDiv+Encoder/dense_4/ActivityRegularizer/mul:z:0,Encoder/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Х
%Decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0•
Decoder/dense_5/MatMulMatMul!Encoder/dense_4/Elu:activations:0-Decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јУ
&Decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0І
Decoder/dense_5/BiasAddBiasAdd Decoder/dense_5/MatMul:product:0.Decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јo
Decoder/dense_5/EluElu Decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јv
Decoder/reshape_2/ShapeShape!Decoder/dense_5/Elu:activations:0*
T0*
_output_shapes
::нѕo
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
valueB:Ђ
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
value	B :c
!Decoder/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`Г
Decoder/reshape_2/Reshape/shapePack(Decoder/reshape_2/strided_slice:output:0*Decoder/reshape_2/Reshape/shape/1:output:0*Decoder/reshape_2/Reshape/shape/2:output:0*Decoder/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
Decoder/reshape_2/ReshapeReshape!Decoder/dense_5/Elu:activations:0(Decoder/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`А
 Decoder/conv2d_transpose_5/ShapeShape"Decoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
::нѕx
.Decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(Decoder/conv2d_transpose_5/strided_sliceStridedSlice)Decoder/conv2d_transpose_5/Shape:output:07Decoder/conv2d_transpose_5/strided_slice/stack:output:09Decoder/conv2d_transpose_5/strided_slice/stack_1:output:09Decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`Р
 Decoder/conv2d_transpose_5/stackPack1Decoder/conv2d_transpose_5/strided_slice:output:0+Decoder/conv2d_transpose_5/stack/1:output:0+Decoder/conv2d_transpose_5/stack/2:output:0+Decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
*Decoder/conv2d_transpose_5/strided_slice_1StridedSlice)Decoder/conv2d_transpose_5/stack:output:09Decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0Ј
+Decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_5/stack:output:0BDecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ў
"Decoder/conv2d_transpose_5/BiasAddBiasAdd4Decoder/conv2d_transpose_5/conv2d_transpose:output:09Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`М
Decoder/conv2d_transpose_5/EluElu+Decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`n
Decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_5/mulMul&Decoder/up_sampling2d_5/Const:output:0(Decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:р
4Decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_5/Elu:activations:0Decoder/up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€`*
half_pixel_centers(£
 Decoder/conv2d_transpose_6/ShapeShapeEDecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕx
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
valueB:Ў
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
value	B :d
"Decoder/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Р
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
valueB:а
*Decoder/conv2d_transpose_6/strided_slice_1StridedSlice)Decoder/conv2d_transpose_6/stack:output:09Decoder/conv2d_transpose_6/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0Џ
+Decoder/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_6/stack:output:0BDecoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
"Decoder/conv2d_transpose_6/BiasAddBiasAdd4Decoder/conv2d_transpose_6/conv2d_transpose:output:09Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@М
Decoder/conv2d_transpose_6/EluElu+Decoder/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@n
Decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
Decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_6/mulMul&Decoder/up_sampling2d_6/Const:output:0(Decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:р
4Decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_6/Elu:activations:0Decoder/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
half_pixel_centers(£
 Decoder/conv2d_transpose_7/ShapeShapeEDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕx
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
valueB:Ў
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
value	B : d
"Decoder/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"Decoder/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Р
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
valueB:а
*Decoder/conv2d_transpose_7/strided_slice_1StridedSlice)Decoder/conv2d_transpose_7/stack:output:09Decoder/conv2d_transpose_7/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask∆
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Џ
+Decoder/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_7/stack:output:0BDecoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0EDecoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
paddingSAME*
strides
®
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
"Decoder/conv2d_transpose_7/BiasAddBiasAdd4Decoder/conv2d_transpose_7/conv2d_transpose:output:09Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @М
Decoder/conv2d_transpose_7/EluElu+Decoder/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @n
Decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   p
Decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Щ
Decoder/up_sampling2d_7/mulMul&Decoder/up_sampling2d_7/Const:output:0(Decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:с
4Decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_7/Elu:activations:0Decoder/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(o
Decoder/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   €
(Decoder/resizing_2/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_2/resize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(С
IdentityIdentity9Decoder/resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аo

Identity_1Identity/Encoder/dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: і
NoOpNoOp2^Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp'^Decoder/dense_5/BiasAdd/ReadVariableOp&^Decoder/dense_5/MatMul/ReadVariableOp(^Encoder/conv2d_5/BiasAdd/ReadVariableOp'^Encoder/conv2d_5/Conv2D/ReadVariableOp(^Encoder/conv2d_6/BiasAdd/ReadVariableOp'^Encoder/conv2d_6/Conv2D/ReadVariableOp(^Encoder/conv2d_7/BiasAdd/ReadVariableOp'^Encoder/conv2d_7/Conv2D/ReadVariableOp'^Encoder/dense_4/BiasAdd/ReadVariableOp&^Encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2f
1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_5/BiasAdd/ReadVariableOp&Decoder/dense_5/BiasAdd/ReadVariableOp2N
%Decoder/dense_5/MatMul/ReadVariableOp%Decoder/dense_5/MatMul/ReadVariableOp2R
'Encoder/conv2d_5/BiasAdd/ReadVariableOp'Encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_5/Conv2D/ReadVariableOp&Encoder/conv2d_5/Conv2D/ReadVariableOp2R
'Encoder/conv2d_6/BiasAdd/ReadVariableOp'Encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_6/Conv2D/ReadVariableOp&Encoder/conv2d_6/Conv2D/ReadVariableOp2R
'Encoder/conv2d_7/BiasAdd/ReadVariableOp'Encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_7/Conv2D/ReadVariableOp&Encoder/conv2d_7/Conv2D/ReadVariableOp2P
&Encoder/dense_4/BiasAdd/ReadVariableOp&Encoder/dense_4/BiasAdd/ReadVariableOp2N
%Encoder/dense_4/MatMul/ReadVariableOp%Encoder/dense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954

inputsB
(conv2d_transpose_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
value	B :`y
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
№

c
D__inference_dropout_layer_call_and_return_conditional_losses_2271503

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€  *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€  i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Г
ю
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
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
:€€€€€€€€€ @ V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ @ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€@А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
н	
”
)__inference_Decoder_layer_call_fn_2272255
input_8
unknown:	 ј
	unknown_0:	ј#
	unknown_1:``
	unknown_2:`#
	unknown_3:@`
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272236x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€ 
!
_user_specified_name	input_8
®
Ц
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272493

inputs)
encoder_2272456: 
encoder_2272458: )
encoder_2272460: @
encoder_2272462:@)
encoder_2272464:@`
encoder_2272466:`"
encoder_2272468:	ј 
encoder_2272470: "
decoder_2272474:	 ј
decoder_2272476:	ј)
decoder_2272478:``
decoder_2272480:`)
decoder_2272482:@`
decoder_2272484:@)
decoder_2272486:@
decoder_2272488:
identity

identity_1ИҐDecoder/StatefulPartitionedCallҐEncoder/StatefulPartitionedCallд
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_2272456encoder_2272458encoder_2272460encoder_2272462encoder_2272464encoder_2272466encoder_2272468encoder_2272470*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271702М
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_2272474decoder_2272476decoder_2272478decoder_2272480decoder_2272482decoder_2272484decoder_2272486decoder_2272488*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272236А
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аh

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: К
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
Ь

ц
D__inference_dense_4_layer_call_and_return_conditional_losses_2273729

inputs1
matmul_readvariableop_resource:	ј -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
®
Ц
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272571

inputs)
encoder_2272534: 
encoder_2272536: )
encoder_2272538: @
encoder_2272540:@)
encoder_2272542:@`
encoder_2272544:`"
encoder_2272546:	ј 
encoder_2272548: "
decoder_2272552:	 ј
decoder_2272554:	ј)
decoder_2272556:``
decoder_2272558:`)
decoder_2272560:@`
decoder_2272562:@)
decoder_2272564:@
decoder_2272566:
identity

identity_1ИҐDecoder/StatefulPartitionedCallҐEncoder/StatefulPartitionedCallд
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_2272534encoder_2272536encoder_2272538encoder_2272540encoder_2272542encoder_2272544encoder_2272546encoder_2272548*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271764М
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_2272552decoder_2272554decoder_2272556decoder_2272558decoder_2272560decoder_2272562decoder_2272564decoder_2272566*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272286А
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аh

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: К
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
Б
ю
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2273593

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Ґ
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:љ
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
њт
м$
#__inference__traced_restore_2274505
file_prefix:
 assignvariableop_conv2d_5_kernel: .
 assignvariableop_1_conv2d_5_bias: <
"assignvariableop_2_conv2d_6_kernel: @.
 assignvariableop_3_conv2d_6_bias:@<
"assignvariableop_4_conv2d_7_kernel:@`.
 assignvariableop_5_conv2d_7_bias:`4
!assignvariableop_6_dense_4_kernel:	ј -
assignvariableop_7_dense_4_bias: 4
!assignvariableop_8_dense_5_kernel:	 ј.
assignvariableop_9_dense_5_bias:	јG
-assignvariableop_10_conv2d_transpose_5_kernel:``9
+assignvariableop_11_conv2d_transpose_5_bias:`G
-assignvariableop_12_conv2d_transpose_6_kernel:@`9
+assignvariableop_13_conv2d_transpose_6_bias:@G
-assignvariableop_14_conv2d_transpose_7_kernel:@9
+assignvariableop_15_conv2d_transpose_7_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: #
assignvariableop_23_total: #
assignvariableop_24_count: D
*assignvariableop_25_adam_conv2d_5_kernel_m: 6
(assignvariableop_26_adam_conv2d_5_bias_m: D
*assignvariableop_27_adam_conv2d_6_kernel_m: @6
(assignvariableop_28_adam_conv2d_6_bias_m:@D
*assignvariableop_29_adam_conv2d_7_kernel_m:@`6
(assignvariableop_30_adam_conv2d_7_bias_m:`<
)assignvariableop_31_adam_dense_4_kernel_m:	ј 5
'assignvariableop_32_adam_dense_4_bias_m: <
)assignvariableop_33_adam_dense_5_kernel_m:	 ј6
'assignvariableop_34_adam_dense_5_bias_m:	јN
4assignvariableop_35_adam_conv2d_transpose_5_kernel_m:``@
2assignvariableop_36_adam_conv2d_transpose_5_bias_m:`N
4assignvariableop_37_adam_conv2d_transpose_6_kernel_m:@`@
2assignvariableop_38_adam_conv2d_transpose_6_bias_m:@N
4assignvariableop_39_adam_conv2d_transpose_7_kernel_m:@@
2assignvariableop_40_adam_conv2d_transpose_7_bias_m:D
*assignvariableop_41_adam_conv2d_5_kernel_v: 6
(assignvariableop_42_adam_conv2d_5_bias_v: D
*assignvariableop_43_adam_conv2d_6_kernel_v: @6
(assignvariableop_44_adam_conv2d_6_bias_v:@D
*assignvariableop_45_adam_conv2d_7_kernel_v:@`6
(assignvariableop_46_adam_conv2d_7_bias_v:`<
)assignvariableop_47_adam_dense_4_kernel_v:	ј 5
'assignvariableop_48_adam_dense_4_bias_v: <
)assignvariableop_49_adam_dense_5_kernel_v:	 ј6
'assignvariableop_50_adam_dense_5_bias_v:	јN
4assignvariableop_51_adam_conv2d_transpose_5_kernel_v:``@
2assignvariableop_52_adam_conv2d_transpose_5_bias_v:`N
4assignvariableop_53_adam_conv2d_transpose_6_kernel_v:@`@
2assignvariableop_54_adam_conv2d_transpose_6_bias_v:@N
4assignvariableop_55_adam_conv2d_transpose_7_kernel_v:@@
2assignvariableop_56_adam_conv2d_transpose_7_bias_v:
identity_58ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9№
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*В
valueшBх:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*З
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B √
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapesл
и::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_5_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_5_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_6_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_6_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_7_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_7_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_5_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_5_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_6_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_6_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_7_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_7_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_4_kernel_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_4_bias_mIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_5_kernel_mIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_5_bias_mIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_conv2d_transpose_5_kernel_mIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_conv2d_transpose_5_bias_mIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_conv2d_transpose_6_kernel_mIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_conv2d_transpose_6_bias_mIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_conv2d_transpose_7_kernel_mIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_conv2d_transpose_7_bias_mIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_5_kernel_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_5_bias_vIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_6_kernel_vIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_6_bias_vIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_7_kernel_vIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_7_bias_vIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_4_kernel_vIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_4_bias_vIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_5_kernel_vIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_5_bias_vIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_5_kernel_vIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_5_bias_vIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_conv2d_transpose_6_kernel_vIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_conv2d_transpose_6_bias_vIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_conv2d_transpose_7_kernel_vIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_conv2d_transpose_7_bias_vIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 µ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: Ґ

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*З
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
≥
G
+__inference_reshape_2_layer_call_fn_2273754

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
У
d
+__inference_dropout_2_layer_call_fn_2273665

inputs
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271567w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
Ї
M
1__inference_max_pooling2d_6_layer_call_fn_2273598

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2271438Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
љ
E
)__inference_dropout_layer_call_fn_2273556

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271617h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
У
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2273603

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
ю
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2273650

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
м	
—
)__inference_Encoder_layer_call_fn_2273135

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
ё

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271535

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕ†
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Ѓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
щ
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271641

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€`c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
Ѕ
G
+__inference_dropout_1_layer_call_fn_2273613

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271629h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
пw
Е
D__inference_Decoder_layer_call_and_return_conditional_losses_2273516

inputs9
&dense_5_matmul_readvariableop_resource:	 ј6
'dense_5_biasadd_readvariableop_resource:	јU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:``@
2conv2d_transpose_5_biasadd_readvariableop_resource:`U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@`@
2conv2d_transpose_6_biasadd_readvariableop_resource:@U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityИҐ)conv2d_transpose_5/BiasAdd/ReadVariableOpҐ2conv2d_transpose_5/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_6/BiasAdd/ReadVariableOpҐ2conv2d_transpose_6/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_7/BiasAdd/ReadVariableOpҐ2conv2d_transpose_7/conv2d_transpose/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јГ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0П
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ј_
dense_5/EluEludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јf
reshape_2/ShapeShapedense_5/Elu:activations:0*
T0*
_output_shapes
::нѕg
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
valueB:Г
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
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`џ
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:У
reshape_2/ReshapeReshapedense_5/Elu:activations:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`p
conv2d_transpose_5/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
::нѕp
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`и
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0Ч
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0reshape_2/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Ш
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ј
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`|
conv2d_transpose_5/EluElu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`f
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:Ў
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_5/Elu:activations:0up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€`*
half_pixel_centers(У
conv2d_transpose_6/ShapeShape=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕp
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
valueB:∞
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
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@и
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
valueB:Є
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0Ї
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Ш
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ј
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@|
conv2d_transpose_6/EluElu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:Ў
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_6/Elu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
half_pixel_centers(У
conv2d_transpose_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕp
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
valueB:∞
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
value	B : \
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :и
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
valueB:Є
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ї
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
paddingSAME*
strides
Ш
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @|
conv2d_transpose_7/EluElu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:ў
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_7/Elu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(g
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   з
 resizing_2/resize/ResizeBilinearResizeBilinear=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0resizing_2/resize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(Й
IdentityIdentity1resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А™
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2273871

inputsB
(conv2d_transpose_readvariableop_resource:@`-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
value	B :@y
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
ш(
с
D__inference_Decoder_layer_call_and_return_conditional_losses_2272175
input_8"
dense_5_2272127:	 ј
dense_5_2272129:	ј4
conv2d_transpose_5_2272148:``(
conv2d_transpose_5_2272150:`4
conv2d_transpose_6_2272154:@`(
conv2d_transpose_6_2272156:@4
conv2d_transpose_7_2272160:@(
conv2d_transpose_7_2272162:
identityИҐ*conv2d_transpose_5/StatefulPartitionedCallҐ*conv2d_transpose_6/StatefulPartitionedCallҐ*conv2d_transpose_7/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallс
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_5_2272127dense_5_2272129*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126е
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146њ
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_5_2272148conv2d_transpose_5_2272150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954О
up_sampling2d_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977„
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_transpose_6_2272154conv2d_transpose_6_2272156*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018О
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041„
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_2272160conv2d_transpose_7_2272162*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082О
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105и
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Ап
NoOpNoOp+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€ 
!
_user_specified_name	input_8
х(
р
D__inference_Decoder_layer_call_and_return_conditional_losses_2272286

inputs"
dense_5_2272260:	 ј
dense_5_2272262:	ј4
conv2d_transpose_5_2272266:``(
conv2d_transpose_5_2272268:`4
conv2d_transpose_6_2272272:@`(
conv2d_transpose_6_2272274:@4
conv2d_transpose_7_2272278:@(
conv2d_transpose_7_2272280:
identityИҐ*conv2d_transpose_5/StatefulPartitionedCallҐ*conv2d_transpose_6/StatefulPartitionedCallҐ*conv2d_transpose_7/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallр
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_2272260dense_5_2272262*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126е
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2272146њ
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_transpose_5_2272266conv2d_transpose_5_2272268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954О
up_sampling2d_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2271977„
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_transpose_6_2272272conv2d_transpose_6_2272274*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018О
up_sampling2d_6/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2272041„
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_7_2272278conv2d_transpose_7_2272280*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2272082О
up_sampling2d_7/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105и
resizing_2/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172{
IdentityIdentity#resizing_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Ап
NoOpNoOp+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
…
©
4__inference_conv2d_transpose_5_layer_call_fn_2273777

inputs!
unknown:``
	unknown_0:`
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2271954Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
м
Я
*__inference_conv2d_6_layer_call_fn_2273582

inputs!
unknown: @
	unknown_0:@
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2271516w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2273931

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
о
Я
*__inference_conv2d_5_layer_call_fn_2273525

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ @ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2271484w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ @ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€@А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
ѕ
c
G__inference_resizing_2_layer_call_and_return_conditional_losses_2272172

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   Ъ
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€@А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
d
+__inference_dropout_1_layer_call_fn_2273608

inputs
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271535w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
пw
Е
D__inference_Decoder_layer_call_and_return_conditional_losses_2273421

inputs9
&dense_5_matmul_readvariableop_resource:	 ј6
'dense_5_biasadd_readvariableop_resource:	јU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:``@
2conv2d_transpose_5_biasadd_readvariableop_resource:`U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@`@
2conv2d_transpose_6_biasadd_readvariableop_resource:@U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityИҐ)conv2d_transpose_5/BiasAdd/ReadVariableOpҐ2conv2d_transpose_5/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_6/BiasAdd/ReadVariableOpҐ2conv2d_transpose_6/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_7/BiasAdd/ReadVariableOpҐ2conv2d_transpose_7/conv2d_transpose/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ј*
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€јГ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0П
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ј_
dense_5/EluEludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€јf
reshape_2/ShapeShapedense_5/Elu:activations:0*
T0*
_output_shapes
::нѕg
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
valueB:Г
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
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`џ
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:У
reshape_2/ReshapeReshapedense_5/Elu:activations:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`p
conv2d_transpose_5/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
::нѕp
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`и
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0Ч
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0reshape_2/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Ш
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ј
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`|
conv2d_transpose_5/EluElu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`f
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:Ў
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_5/Elu:activations:0up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€`*
half_pixel_centers(У
conv2d_transpose_6/ShapeShape=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕp
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
valueB:∞
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
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@и
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
valueB:Є
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0Ї
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Ш
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ј
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@|
conv2d_transpose_6/EluElu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:Ў
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_6/Elu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
half_pixel_centers(У
conv2d_transpose_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
::нѕp
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
valueB:∞
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
value	B : \
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :и
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
valueB:Є
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskґ
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ї
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:€€€€€€€€€ @*
paddingSAME*
strides
Ш
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @|
conv2d_transpose_7/EluElu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Б
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:ў
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_7/Elu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(g
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   А   з
 resizing_2/resize/ResizeBilinearResizeBilinear=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0resizing_2/resize/size:output:0*
T0*0
_output_shapes
:€€€€€€€€€@А*
half_pixel_centers(Й
IdentityIdentity1resizing_2/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А™
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€ : : : : : : : : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
¶>
а
D__inference_Encoder_layer_call_and_return_conditional_losses_2273284

inputsA
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@`6
(conv2d_7_biasadd_readvariableop_resource:`9
&dense_4_matmul_readvariableop_resource:	ј 5
'dense_4_biasadd_readvariableop_resource: 
identity

identity_1ИҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpО
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ђ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ *
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ @ h
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ @ ™
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Elu:activations:0*/
_output_shapes
:€€€€€€€€€  *
ksize
*
paddingSAME*
strides
x
dropout/IdentityIdentity max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€  О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Њ
conv2d_6/Conv2DConv2Ddropout/Identity:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@h
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@™
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Elu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
z
dropout_1/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@О
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ј
conv2d_7/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`*
paddingSAME*
strides
Д
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ш
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`h
conv2d_7/EluEluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€`™
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Elu:activations:0*/
_output_shapes
:€€€€€€€€€`*
ksize
*
paddingSAME*
strides
z
dropout_2/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€``
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   Ж
flatten_2/ReshapeReshapedropout_2/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€јЕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	ј *
dtype0Н
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
dense_4/EluEludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ s
dense_4/ActivityRegularizer/AbsAbsdense_4/Elu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ r
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/Abs:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: x
!dense_4/ActivityRegularizer/ShapeShapedense_4/Elu:activations:0*
T0*
_output_shapes
::нѕy
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
valueB:Ё
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ъ
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: h
IdentityIdentitydense_4/Elu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ g

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: –
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
Ђ
Ч
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272410
input_9)
encoder_2272373: 
encoder_2272375: )
encoder_2272377: @
encoder_2272379:@)
encoder_2272381:@`
encoder_2272383:`"
encoder_2272385:	ј 
encoder_2272387: "
decoder_2272391:	 ј
decoder_2272393:	ј)
decoder_2272395:``
decoder_2272397:`)
decoder_2272399:@`
decoder_2272401:@)
decoder_2272403:@
decoder_2272405:
identity

identity_1ИҐDecoder/StatefulPartitionedCallҐEncoder/StatefulPartitionedCallе
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_2272373encoder_2272375encoder_2272377encoder_2272379encoder_2272381encoder_2272383encoder_2272385encoder_2272387*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271702М
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_2272391decoder_2272393decoder_2272395decoder_2272397decoder_2272399decoder_2272401decoder_2272403decoder_2272405*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272236А
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аh

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: К
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
щ
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273630

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
п	
“
)__inference_Encoder_layer_call_fn_2271722
input_7!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_7
∆
Ш
)__inference_dense_5_layer_call_fn_2273738

inputs
unknown:	 ј
	unknown_0:	ј
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2272126p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ
G
+__inference_dropout_2_layer_call_fn_2273670

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_2271641h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
™
д
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272804

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
	unknown_7:	 ј
	unknown_8:	ј#
	unknown_9:``

unknown_10:`$

unknown_11:@`

unknown_12:@$

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallЉ
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
:€€€€€€€€€@А: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *i
fdRb
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272571x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
щ
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2271629

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
Ч
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272450
input_9)
encoder_2272413: 
encoder_2272415: )
encoder_2272417: @
encoder_2272419:@)
encoder_2272421:@`
encoder_2272423:`"
encoder_2272425:	ј 
encoder_2272427: "
decoder_2272431:	 ј
decoder_2272433:	ј)
decoder_2272435:``
decoder_2272437:`)
decoder_2272439:@`
decoder_2272441:@)
decoder_2272443:@
decoder_2272445:
identity

identity_1ИҐDecoder/StatefulPartitionedCallҐEncoder/StatefulPartitionedCallе
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_2272413encoder_2272415encoder_2272417encoder_2272419encoder_2272421encoder_2272423encoder_2272425encoder_2272427*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271764М
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_2272431decoder_2272433decoder_2272435decoder_2272437decoder_2272439decoder_2272441decoder_2272443decoder_2272445*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€@А**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_2272286А
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€@Аh

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: К
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€@А: : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:€€€€€€€€€@А
!
_user_specified_name	input_9
У
h
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2273546

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ш
∆
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2273718

inputs
unknown:	ј 
	unknown_0: 
identity

identity_1ИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2271588§
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
 *-
config_proto

CPU

GPU 2J 8В *9
f4R2
0__inference_dense_4_activity_regularizer_2271469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ X

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
»
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_2271575

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ј   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€јY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€`:W S
/
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
м	
—
)__inference_Encoder_layer_call_fn_2273157

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@`
	unknown_4:`
	unknown_5:	ј 
	unknown_6: 
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€ : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_2271764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€@А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€@А
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2273811

inputsB
(conv2d_transpose_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
value	B :`y
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:``*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
Ї
M
1__inference_up_sampling2d_7_layer_call_fn_2273936

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2272105Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–!
Ь
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2272018

inputsB
(conv2d_transpose_readvariableop_resource:@`-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
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
valueB:—
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
valueB:ў
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
valueB:ў
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
value	B :@y
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
valueB:ў
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype0№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
П
b
)__inference_dropout_layer_call_fn_2273551

inputs
identityИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2271503w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs"у
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Љ
serving_default®
D
input_99
serving_default_input_9:0€€€€€€€€€@АD
Decoder9
StatefulPartitionedCall:0€€€€€€€€€@Аtensorflow/serving/predict:пґ
г
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
«
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
# _self_saveable_object_factories"
_tf_keras_sequential
ђ
!layer_with_weights-0
!layer-0
"layer-1
#layer_with_weights-1
#layer-2
$layer-3
%layer_with_weights-2
%layer-4
&layer-5
'layer_with_weights-3
'layer-6
(layer-7
)layer-8
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
#0_self_saveable_object_factories"
_tf_keras_sequential
Ц
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15"
trackable_list_wrapper
Ц
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15"
trackable_list_wrapper
 "
trackable_list_wrapper
 
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
њ
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32‘
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272529
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272607
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272766
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272804µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
Ђ
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32ј
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272410
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272450
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272969
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2273113µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
ЌB 
"__inference__wrapped_model_2271420input_9"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
У
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_rate1mІ2m®3m©4m™5mЂ6mђ7m≠8mЃ9mѓ:m∞;m±<m≤=m≥>mі?mµ@mґ1vЈ2vЄ3vє4vЇ5vї6vЉ7vљ8vЊ9vњ:vј;vЅ<v¬=v√>vƒ?v≈@v∆"
	optimizer
,
Sserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
В
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

1kernel
2bias
#Z_self_saveable_object_factories
 [_jit_compiled_convolution_op"
_tf_keras_layer
 
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
#b_self_saveable_object_factories"
_tf_keras_layer
б
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
i_random_generator
#j_self_saveable_object_factories"
_tf_keras_layer
В
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

3kernel
4bias
#q_self_saveable_object_factories
 r_jit_compiled_convolution_op"
_tf_keras_layer
 
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories"
_tf_keras_layer
г
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
А_random_generator
$Б_self_saveable_object_factories"
_tf_keras_layer
К
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses

5kernel
6bias
$И_self_saveable_object_factories
!Й_jit_compiled_convolution_op"
_tf_keras_layer
—
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses
$Р_self_saveable_object_factories"
_tf_keras_layer
й
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator
$Ш_self_saveable_object_factories"
_tf_keras_layer
—
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
$Я_self_saveable_object_factories"
_tf_keras_layer
з
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses

7kernel
8bias
$¶_self_saveable_object_factories"
_tf_keras_layer
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
„
ђtrace_0
≠trace_1
Ѓtrace_2
ѓtrace_32д
)__inference_Encoder_layer_call_fn_2271722
)__inference_Encoder_layer_call_fn_2271784
)__inference_Encoder_layer_call_fn_2273135
)__inference_Encoder_layer_call_fn_2273157µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0z≠trace_1zЃtrace_2zѓtrace_3
√
∞trace_0
±trace_1
≤trace_2
≥trace_32–
D__inference_Encoder_layer_call_and_return_conditional_losses_2271604
D__inference_Encoder_layer_call_and_return_conditional_losses_2271659
D__inference_Encoder_layer_call_and_return_conditional_losses_2273231
D__inference_Encoder_layer_call_and_return_conditional_losses_2273284µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0z±trace_1z≤trace_2z≥trace_3
 "
trackable_dict_wrapper
з
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
Є__call__
+є&call_and_return_all_conditional_losses

9kernel
:bias
$Ї_self_saveable_object_factories"
_tf_keras_layer
—
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses
$Ѕ_self_saveable_object_factories"
_tf_keras_layer
К
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses

;kernel
<bias
$»_self_saveable_object_factories
!…_jit_compiled_convolution_op"
_tf_keras_layer
—
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
$–_self_saveable_object_factories"
_tf_keras_layer
К
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses

=kernel
>bias
$„_self_saveable_object_factories
!Ў_jit_compiled_convolution_op"
_tf_keras_layer
—
ў	variables
Џtrainable_variables
џregularization_losses
№	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses
$я_self_saveable_object_factories"
_tf_keras_layer
К
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

?kernel
@bias
$ж_self_saveable_object_factories
!з_jit_compiled_convolution_op"
_tf_keras_layer
—
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses
$о_self_saveable_object_factories"
_tf_keras_layer
—
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
$х_self_saveable_object_factories"
_tf_keras_layer
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
„
ыtrace_0
ьtrace_1
эtrace_2
юtrace_32д
)__inference_Decoder_layer_call_fn_2272255
)__inference_Decoder_layer_call_fn_2272305
)__inference_Decoder_layer_call_fn_2273305
)__inference_Decoder_layer_call_fn_2273326µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zыtrace_0zьtrace_1zэtrace_2zюtrace_3
√
€trace_0
Аtrace_1
Бtrace_2
Вtrace_32–
D__inference_Decoder_layer_call_and_return_conditional_losses_2272175
D__inference_Decoder_layer_call_and_return_conditional_losses_2272204
D__inference_Decoder_layer_call_and_return_conditional_losses_2273421
D__inference_Decoder_layer_call_and_return_conditional_losses_2273516µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z€trace_0zАtrace_1zБtrace_2zВtrace_3
 "
trackable_dict_wrapper
):' 2conv2d_5/kernel
: 2conv2d_5/bias
):' @2conv2d_6/kernel
:@2conv2d_6/bias
):'@`2conv2d_7/kernel
:`2conv2d_7/bias
!:	ј 2dense_4/kernel
: 2dense_4/bias
!:	 ј2dense_5/kernel
:ј2dense_5/bias
3:1``2conv2d_transpose_5/kernel
%:#`2conv2d_transpose_5/bias
3:1@`2conv2d_transpose_6/kernel
%:#@2conv2d_transpose_6/bias
3:1@2conv2d_transpose_7/kernel
%:#2conv2d_transpose_7/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
НBК
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272529input_9"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272607input_9"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272766inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272804inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®B•
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272410input_9"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®B•
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272450input_9"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272969inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2273113inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ћB…
%__inference_signature_wrapper_2272728input_9"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
≤
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
ж
Кtrace_02«
*__inference_conv2d_5_layer_call_fn_2273525Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zКtrace_0
Б
Лtrace_02в
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2273536Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЛtrace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
н
Сtrace_02ќ
1__inference_max_pooling2d_5_layer_call_fn_2273541Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zСtrace_0
И
Тtrace_02й
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2273546Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zТtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
љ
Шtrace_0
Щtrace_12В
)__inference_dropout_layer_call_fn_2273551
)__inference_dropout_layer_call_fn_2273556©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zШtrace_0zЩtrace_1
у
Ъtrace_0
Ыtrace_12Є
D__inference_dropout_layer_call_and_return_conditional_losses_2273568
D__inference_dropout_layer_call_and_return_conditional_losses_2273573©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0zЫtrace_1
D
$Ь_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ж
Ґtrace_02«
*__inference_conv2d_6_layer_call_fn_2273582Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0
Б
£trace_02в
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2273593Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
н
©trace_02ќ
1__inference_max_pooling2d_6_layer_call_fn_2273598Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z©trace_0
И
™trace_02й
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2273603Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ѕ
∞trace_0
±trace_12Ж
+__inference_dropout_1_layer_call_fn_2273608
+__inference_dropout_1_layer_call_fn_2273613©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0z±trace_1
ч
≤trace_0
≥trace_12Љ
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273625
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273630©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0z≥trace_1
D
$і_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
ж
Їtrace_02«
*__inference_conv2d_7_layer_call_fn_2273639Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
Б
їtrace_02в
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2273650Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zїtrace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
н
Ѕtrace_02ќ
1__inference_max_pooling2d_7_layer_call_fn_2273655Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
И
¬trace_02й
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2273660Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¬trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
Ѕ
»trace_0
…trace_12Ж
+__inference_dropout_2_layer_call_fn_2273665
+__inference_dropout_2_layer_call_fn_2273670©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0z…trace_1
ч
 trace_0
Ћtrace_12Љ
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273682
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273687©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z trace_0zЋtrace_1
D
$ћ_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
з
“trace_02»
+__inference_flatten_2_layer_call_fn_2273692Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z“trace_0
В
”trace_02г
F__inference_flatten_2_layer_call_and_return_conditional_losses_2273698Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z”trace_0
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
÷
‘non_trainable_variables
’layers
÷metrics
 „layer_regularization_losses
Ўlayer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
ўactivity_regularizer_fn
+•&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
е
џtrace_02∆
)__inference_dense_4_layer_call_fn_2273707Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zџtrace_0
Д
№trace_02е
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2273718Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z№trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
)__inference_Encoder_layer_call_fn_2271722input_7"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
)__inference_Encoder_layer_call_fn_2271784input_7"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
)__inference_Encoder_layer_call_fn_2273135inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
)__inference_Encoder_layer_call_fn_2273157inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
D__inference_Encoder_layer_call_and_return_conditional_losses_2271604input_7"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
D__inference_Encoder_layer_call_and_return_conditional_losses_2271659input_7"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЛBИ
D__inference_Encoder_layer_call_and_return_conditional_losses_2273231inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЛBИ
D__inference_Encoder_layer_call_and_return_conditional_losses_2273284inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ёnon_trainable_variables
ёlayers
яmetrics
 аlayer_regularization_losses
бlayer_metrics
і	variables
µtrainable_variables
ґregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
е
вtrace_02∆
)__inference_dense_5_layer_call_fn_2273738Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zвtrace_0
А
гtrace_02б
D__inference_dense_5_layer_call_and_return_conditional_losses_2273749Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zгtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
з
йtrace_02»
+__inference_reshape_2_layer_call_fn_2273754Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zйtrace_0
В
кtrace_02г
F__inference_reshape_2_layer_call_and_return_conditional_losses_2273768Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zкtrace_0
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
р
рtrace_02—
4__inference_conv2d_transpose_5_layer_call_fn_2273777Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zрtrace_0
Л
сtrace_02м
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2273811Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zсtrace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
н
чtrace_02ќ
1__inference_up_sampling2d_5_layer_call_fn_2273816Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zчtrace_0
И
шtrace_02й
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2273828Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zшtrace_0
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
Є
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
р
юtrace_02—
4__inference_conv2d_transpose_6_layer_call_fn_2273837Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zюtrace_0
Л
€trace_02м
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2273871Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z€trace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
ў	variables
Џtrainable_variables
џregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
н
Еtrace_02ќ
1__inference_up_sampling2d_6_layer_call_fn_2273876Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЕtrace_0
И
Жtrace_02й
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2273888Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЖtrace_0
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
р
Мtrace_02—
4__inference_conv2d_transpose_7_layer_call_fn_2273897Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0
Л
Нtrace_02м
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2273931Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zНtrace_0
 "
trackable_dict_wrapper
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
н
Уtrace_02ќ
1__inference_up_sampling2d_7_layer_call_fn_2273936Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zУtrace_0
И
Фtrace_02й
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2273948Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zФtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
и
Ъtrace_02…
,__inference_resizing_2_layer_call_fn_2273953Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0
Г
Ыtrace_02д
G__inference_resizing_2_layer_call_and_return_conditional_losses_2273959Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЫtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
!0
"1
#2
$3
%4
&5
'6
(7
)8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
)__inference_Decoder_layer_call_fn_2272255input_8"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
)__inference_Decoder_layer_call_fn_2272305input_8"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
)__inference_Decoder_layer_call_fn_2273305inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
)__inference_Decoder_layer_call_fn_2273326inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
D__inference_Decoder_layer_call_and_return_conditional_losses_2272175input_8"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
D__inference_Decoder_layer_call_and_return_conditional_losses_2272204input_8"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЛBИ
D__inference_Decoder_layer_call_and_return_conditional_losses_2273421inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЛBИ
D__inference_Decoder_layer_call_and_return_conditional_losses_2273516inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
Ь	variables
Э	keras_api

Юtotal

Яcount"
_tf_keras_metric
c
†	variables
°	keras_api

Ґtotal

£count
§
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
‘B—
*__inference_conv2d_5_layer_call_fn_2273525inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2273536inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_max_pooling2d_5_layer_call_fn_2273541inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2273546inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
дBб
)__inference_dropout_layer_call_fn_2273551inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
дBб
)__inference_dropout_layer_call_fn_2273556inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
D__inference_dropout_layer_call_and_return_conditional_losses_2273568inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
D__inference_dropout_layer_call_and_return_conditional_losses_2273573inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
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
‘B—
*__inference_conv2d_6_layer_call_fn_2273582inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2273593inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_max_pooling2d_6_layer_call_fn_2273598inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2273603inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
жBг
+__inference_dropout_1_layer_call_fn_2273608inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
жBг
+__inference_dropout_1_layer_call_fn_2273613inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273625inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273630inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
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
‘B—
*__inference_conv2d_7_layer_call_fn_2273639inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2273650inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_max_pooling2d_7_layer_call_fn_2273655inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2273660inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
жBг
+__inference_dropout_2_layer_call_fn_2273665inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
жBг
+__inference_dropout_2_layer_call_fn_2273670inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273682inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273687inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
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
’B“
+__inference_flatten_2_layer_call_fn_2273692inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
F__inference_flatten_2_layer_call_and_return_conditional_losses_2273698inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
у
•trace_02‘
0__inference_dense_4_activity_regularizer_2271469Я
М≤И
FullArgSpec
argsЪ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	Кz•trace_0
А
¶trace_02б
D__inference_dense_4_layer_call_and_return_conditional_losses_2273729Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¶trace_0
”B–
)__inference_dense_4_layer_call_fn_2273707inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2273718inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
”B–
)__inference_dense_5_layer_call_fn_2273738inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_dense_5_layer_call_and_return_conditional_losses_2273749inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
’B“
+__inference_reshape_2_layer_call_fn_2273754inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
F__inference_reshape_2_layer_call_and_return_conditional_losses_2273768inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
4__inference_conv2d_transpose_5_layer_call_fn_2273777inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2273811inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_up_sampling2d_5_layer_call_fn_2273816inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2273828inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
4__inference_conv2d_transpose_6_layer_call_fn_2273837inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2273871inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_up_sampling2d_6_layer_call_fn_2273876inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2273888inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
4__inference_conv2d_transpose_7_layer_call_fn_2273897inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2273931inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
џBЎ
1__inference_up_sampling2d_7_layer_call_fn_2273936inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2273948inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_resizing_2_layer_call_fn_2273953inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_resizing_2_layer_call_and_return_conditional_losses_2273959inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Ю0
Я1"
trackable_list_wrapper
.
Ь	variables"
_generic_user_object
:  (2total
:  (2count
0
Ґ0
£1"
trackable_list_wrapper
.
†	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
№Bў
0__inference_dense_4_activity_regularizer_2271469x"Я
М≤И
FullArgSpec
argsЪ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
оBл
D__inference_dense_4_layer_call_and_return_conditional_losses_2273729inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.:, 2Adam/conv2d_5/kernel/m
 : 2Adam/conv2d_5/bias/m
.:, @2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
.:,@`2Adam/conv2d_7/kernel/m
 :`2Adam/conv2d_7/bias/m
&:$	ј 2Adam/dense_4/kernel/m
: 2Adam/dense_4/bias/m
&:$	 ј2Adam/dense_5/kernel/m
 :ј2Adam/dense_5/bias/m
8:6``2 Adam/conv2d_transpose_5/kernel/m
*:(`2Adam/conv2d_transpose_5/bias/m
8:6@`2 Adam/conv2d_transpose_6/kernel/m
*:(@2Adam/conv2d_transpose_6/bias/m
8:6@2 Adam/conv2d_transpose_7/kernel/m
*:(2Adam/conv2d_transpose_7/bias/m
.:, 2Adam/conv2d_5/kernel/v
 : 2Adam/conv2d_5/bias/v
.:, @2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
.:,@`2Adam/conv2d_7/kernel/v
 :`2Adam/conv2d_7/bias/v
&:$	ј 2Adam/dense_4/kernel/v
: 2Adam/dense_4/bias/v
&:$	 ј2Adam/dense_5/kernel/v
 :ј2Adam/dense_5/bias/v
8:6``2 Adam/conv2d_transpose_5/kernel/v
*:(`2Adam/conv2d_transpose_5/bias/v
8:6@`2 Adam/conv2d_transpose_6/kernel/v
*:(@2Adam/conv2d_transpose_6/bias/v
8:6@2 Adam/conv2d_transpose_7/kernel/v
*:(2Adam/conv2d_transpose_7/bias/vЖ
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272410°123456789:;<=>?@AҐ>
7Ґ4
*К'
input_9€€€€€€€€€@А
p

 
™ "JҐG
+К(
tensor_0€€€€€€€€€@А
Ъ
К

tensor_1_0 Ж
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272450°123456789:;<=>?@AҐ>
7Ґ4
*К'
input_9€€€€€€€€€@А
p 

 
™ "JҐG
+К(
tensor_0€€€€€€€€€@А
Ъ
К

tensor_1_0 Е
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2272969†123456789:;<=>?@@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p

 
™ "JҐG
+К(
tensor_0€€€€€€€€€@А
Ъ
К

tensor_1_0 Е
`__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_and_return_conditional_losses_2273113†123456789:;<=>?@@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p 

 
™ "JҐG
+К(
tensor_0€€€€€€€€€@А
Ъ
К

tensor_1_0 Ћ
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272529Б123456789:;<=>?@AҐ>
7Ґ4
*К'
input_9€€€€€€€€€@А
p

 
™ "*К'
unknown€€€€€€€€€@АЋ
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272607Б123456789:;<=>?@AҐ>
7Ґ4
*К'
input_9€€€€€€€€€@А
p 

 
™ "*К'
unknown€€€€€€€€€@А 
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272766А123456789:;<=>?@@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p

 
™ "*К'
unknown€€€€€€€€€@А 
E__inference_AE_Conv_prep_flatten_STFT_Augmented_layer_call_fn_2272804А123456789:;<=>?@@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p 

 
™ "*К'
unknown€€€€€€€€€@А√
D__inference_Decoder_layer_call_and_return_conditional_losses_2272175{9:;<=>?@8Ґ5
.Ґ+
!К
input_8€€€€€€€€€ 
p

 
™ "5Ґ2
+К(
tensor_0€€€€€€€€€@А
Ъ √
D__inference_Decoder_layer_call_and_return_conditional_losses_2272204{9:;<=>?@8Ґ5
.Ґ+
!К
input_8€€€€€€€€€ 
p 

 
™ "5Ґ2
+К(
tensor_0€€€€€€€€€@А
Ъ ¬
D__inference_Decoder_layer_call_and_return_conditional_losses_2273421z9:;<=>?@7Ґ4
-Ґ*
 К
inputs€€€€€€€€€ 
p

 
™ "5Ґ2
+К(
tensor_0€€€€€€€€€@А
Ъ ¬
D__inference_Decoder_layer_call_and_return_conditional_losses_2273516z9:;<=>?@7Ґ4
-Ґ*
 К
inputs€€€€€€€€€ 
p 

 
™ "5Ґ2
+К(
tensor_0€€€€€€€€€@А
Ъ Э
)__inference_Decoder_layer_call_fn_2272255p9:;<=>?@8Ґ5
.Ґ+
!К
input_8€€€€€€€€€ 
p

 
™ "*К'
unknown€€€€€€€€€@АЭ
)__inference_Decoder_layer_call_fn_2272305p9:;<=>?@8Ґ5
.Ґ+
!К
input_8€€€€€€€€€ 
p 

 
™ "*К'
unknown€€€€€€€€€@АЬ
)__inference_Decoder_layer_call_fn_2273305o9:;<=>?@7Ґ4
-Ґ*
 К
inputs€€€€€€€€€ 
p

 
™ "*К'
unknown€€€€€€€€€@АЬ
)__inference_Decoder_layer_call_fn_2273326o9:;<=>?@7Ґ4
-Ґ*
 К
inputs€€€€€€€€€ 
p 

 
™ "*К'
unknown€€€€€€€€€@Аў
D__inference_Encoder_layer_call_and_return_conditional_losses_2271604Р12345678AҐ>
7Ґ4
*К'
input_7€€€€€€€€€@А
p

 
™ "AҐ>
"К
tensor_0€€€€€€€€€ 
Ъ
К

tensor_1_0 ў
D__inference_Encoder_layer_call_and_return_conditional_losses_2271659Р12345678AҐ>
7Ґ4
*К'
input_7€€€€€€€€€@А
p 

 
™ "AҐ>
"К
tensor_0€€€€€€€€€ 
Ъ
К

tensor_1_0 Ў
D__inference_Encoder_layer_call_and_return_conditional_losses_2273231П12345678@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p

 
™ "AҐ>
"К
tensor_0€€€€€€€€€ 
Ъ
К

tensor_1_0 Ў
D__inference_Encoder_layer_call_and_return_conditional_losses_2273284П12345678@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p 

 
™ "AҐ>
"К
tensor_0€€€€€€€€€ 
Ъ
К

tensor_1_0 Э
)__inference_Encoder_layer_call_fn_2271722p12345678AҐ>
7Ґ4
*К'
input_7€€€€€€€€€@А
p

 
™ "!К
unknown€€€€€€€€€ Э
)__inference_Encoder_layer_call_fn_2271784p12345678AҐ>
7Ґ4
*К'
input_7€€€€€€€€€@А
p 

 
™ "!К
unknown€€€€€€€€€ Ь
)__inference_Encoder_layer_call_fn_2273135o12345678@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p

 
™ "!К
unknown€€€€€€€€€ Ь
)__inference_Encoder_layer_call_fn_2273157o12345678@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€@А
p 

 
™ "!К
unknown€€€€€€€€€ ∞
"__inference__wrapped_model_2271420Й123456789:;<=>?@9Ґ6
/Ґ,
*К'
input_9€€€€€€€€€@А
™ ":™7
5
Decoder*К'
decoder€€€€€€€€€@Аљ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2273536t128Ґ5
.Ґ+
)К&
inputs€€€€€€€€€@А
™ "4Ґ1
*К'
tensor_0€€€€€€€€€ @ 
Ъ Ч
*__inference_conv2d_5_layer_call_fn_2273525i128Ґ5
.Ґ+
)К&
inputs€€€€€€€€€@А
™ ")К&
unknown€€€€€€€€€ @ Љ
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2273593s347Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ "4Ґ1
*К'
tensor_0€€€€€€€€€@
Ъ Ц
*__inference_conv2d_6_layer_call_fn_2273582h347Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ ")К&
unknown€€€€€€€€€@Љ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2273650s567Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "4Ґ1
*К'
tensor_0€€€€€€€€€`
Ъ Ц
*__inference_conv2d_7_layer_call_fn_2273639h567Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ ")К&
unknown€€€€€€€€€`л
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2273811Ч;<IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
Ъ ≈
4__inference_conv2d_transpose_5_layer_call_fn_2273777М;<IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€`л
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2273871Ч=>IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≈
4__inference_conv2d_transpose_6_layer_call_fn_2273837М=>IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@л
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2273931Ч?@IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
4__inference_conv2d_transpose_7_layer_call_fn_2273897М?@IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€c
0__inference_dense_4_activity_regularizer_2271469/Ґ
Ґ
К	
x
™ "К
unknown ≈
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2273718y780Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "AҐ>
"К
tensor_0€€€€€€€€€ 
Ъ
К

tensor_1_0 ђ
D__inference_dense_4_layer_call_and_return_conditional_losses_2273729d780Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ Ж
)__inference_dense_4_layer_call_fn_2273707Y780Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "!К
unknown€€€€€€€€€ ђ
D__inference_dense_5_layer_call_and_return_conditional_losses_2273749d9:/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€ј
Ъ Ж
)__inference_dense_5_layer_call_fn_2273738Y9:/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ""К
unknown€€€€€€€€€јљ
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273625s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ "4Ґ1
*К'
tensor_0€€€€€€€€€@
Ъ љ
F__inference_dropout_1_layer_call_and_return_conditional_losses_2273630s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ "4Ґ1
*К'
tensor_0€€€€€€€€€@
Ъ Ч
+__inference_dropout_1_layer_call_fn_2273608h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ ")К&
unknown€€€€€€€€€@Ч
+__inference_dropout_1_layer_call_fn_2273613h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ ")К&
unknown€€€€€€€€€@љ
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273682s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`
p
™ "4Ґ1
*К'
tensor_0€€€€€€€€€`
Ъ љ
F__inference_dropout_2_layer_call_and_return_conditional_losses_2273687s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`
p 
™ "4Ґ1
*К'
tensor_0€€€€€€€€€`
Ъ Ч
+__inference_dropout_2_layer_call_fn_2273665h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`
p
™ ")К&
unknown€€€€€€€€€`Ч
+__inference_dropout_2_layer_call_fn_2273670h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`
p 
™ ")К&
unknown€€€€€€€€€`ї
D__inference_dropout_layer_call_and_return_conditional_losses_2273568s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  
p
™ "4Ґ1
*К'
tensor_0€€€€€€€€€  
Ъ ї
D__inference_dropout_layer_call_and_return_conditional_losses_2273573s;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  
p 
™ "4Ґ1
*К'
tensor_0€€€€€€€€€  
Ъ Х
)__inference_dropout_layer_call_fn_2273551h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  
p
™ ")К&
unknown€€€€€€€€€  Х
)__inference_dropout_layer_call_fn_2273556h;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  
p 
™ ")К&
unknown€€€€€€€€€  ≤
F__inference_flatten_2_layer_call_and_return_conditional_losses_2273698h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€`
™ "-Ґ*
#К 
tensor_0€€€€€€€€€ј
Ъ М
+__inference_flatten_2_layer_call_fn_2273692]7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€`
™ ""К
unknown€€€€€€€€€јц
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2273546•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_max_pooling2d_5_layer_call_fn_2273541ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ц
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2273603•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_max_pooling2d_6_layer_call_fn_2273598ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ц
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2273660•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_max_pooling2d_7_layer_call_fn_2273655ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€≤
F__inference_reshape_2_layer_call_and_return_conditional_losses_2273768h0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "4Ґ1
*К'
tensor_0€€€€€€€€€`
Ъ М
+__inference_reshape_2_layer_call_fn_2273754]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ ")К&
unknown€€€€€€€€€`ќ
G__inference_resizing_2_layer_call_and_return_conditional_losses_2273959ВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "5Ґ2
+К(
tensor_0€€€€€€€€€@А
Ъ І
,__inference_resizing_2_layer_call_fn_2273953wIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "*К'
unknown€€€€€€€€€@АЊ
%__inference_signature_wrapper_2272728Ф123456789:;<=>?@DҐA
Ґ 
:™7
5
input_9*К'
input_9€€€€€€€€€@А":™7
5
Decoder*К'
decoder€€€€€€€€€@Ац
L__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_2273828•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_up_sampling2d_5_layer_call_fn_2273816ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ц
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2273888•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_up_sampling2d_6_layer_call_fn_2273876ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ц
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2273948•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ –
1__inference_up_sampling2d_7_layer_call_fn_2273936ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€