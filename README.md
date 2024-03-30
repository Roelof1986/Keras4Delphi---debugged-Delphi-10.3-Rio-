**for CPU:**

1. Install Delphi 10.3 Rio/10.4 Sydney
2. Install Python 3.9 (64-bit) or Python 3.12 (64-bit)
3. Install TensorFlow 2.15.0 or TensorFlow 2.16.1
4. Download and unzip Keras4Delphi - debugged
5. Compile and run Project1.dproj or Project2.dproj in Delphi

**for GPU:**

1. Install Delphi 10.3 Rio/10.4 Sydney
2. Install Python 3.10 (64-bit)
3. Install TensorFlow 2.10.0
4. Download and unzip Keras4Delphi - debugged
5. Compile and run Project1.dproj or Project2.dproj in Delphi

**for GPU first (real-time bug fix):**
1. Install CUDA 11.8 and CUDA 12.2
2. Install CuDNN 8.6
3. Install Python 3.9
4. Install TensorFlow 2.10
5. Copy needed dll files to .exe path (when requested by error message while running Delphi project)

Important note: the function ToDoubleArray I manually added, because there was no such function. This function only works on a one-dimensional array! So you have to reshape a multi-dimensional array: y_test := TNDArray(y_test.reshape[len, 1]);
I will fix this bug ASAP.

**Simple code example**

//------------------------

uses
  Keras,
  Keras.Layers,
  Keras.Models,
  System.Types,
  np.Base,
  np.Api,
  np.Utils,
  np.Models,
  PythonEngine,
  Python.Utils,
  System.IOUtils,
  SysUtils,
  Keras.PreProcessing,
  Windows;

var
x, y, x_test, y_test : TNDarray;

y_out2 : Variant;

xtestarray : TArray<Double>;

i : Longint;

begin

TNumPy.Init(True);

//Load train data

SetLength(xtestarray, 16);

for i := 0 to 16-1 do
  xtestarray[i] := Random;

x := TNumPy.npArray<Double>(xtestarray);

x := TNDArray(x.reshape([8, 2]));

SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
Writeln('x shape: '  + x.shape.ToString);
SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});

y := TNumPy.npArray<Double>( [ 0, {1}0, 1, 0, 1, 0, 1, 0 ] );

//Build functional model
var input  : TKInput := TKInput.Create(tnp_shape.Create([2]));
var hidden1: TBaseLayer  := TDense.Create(32, 'relu').&Set([input]);
var hidden2: TBaseLayer  := TDense.Create(64, 'relu').&Set([hidden1]);
var output : TBaseLayer  := TDense.Create(1,  'sigmoid').&Set([hidden2]);

var model : TModel := TModel.Create ( [ input ] , [ output ]);

//Compile and train

model.Compile('adam', 'binary_crossentropy',['accuracy']);

var batch_size : Integer := 2;
var history: THistory := model.Fit(x, y, @batch_size, {10}500,1);

model.Summary;

var logs := history.HistoryLogs;

//Predict (network output)

SetLength(xtestarray, 16);

for i := 0 to 16-1 do
  xtestarray[i] := Random;

x_test := TNumPy.npArray<Double>(xtestarray);

x_test := TNDArray(x.reshape([8, 2]));

y_test := model.Predict(x_test);

y_out2 := y_test.ToDoubleArray;

for i := 0 to 8-1 do
  Writeln('Network output on ', i, ' : ', y_out2[i]);

//Save model and weights
var json : string := model.ToJson;
TFile.WriteAllText('model.json', json);
model.SaveWeight('model.h5');

//Load model and weight
var loaded_model : TBaseModel := TSequential.ModelFromJson(TFile.ReadAllText('model.json'));
loaded_model.LoadWeight('model.h5');

Readln;

end.

//------------------------


****Code example nr. 2 -- Convolutional Neural Network trained on random data**

program Project2_CONV_test2;

uses
  Keras,
  Keras.Layers,
  Keras.Models,
  System.Types,
  np.Base,
  np.Api,
  np.Utils,
  np.Models,
  PythonEngine,
  Python.Utils,
  System.IOUtils,
  SysUtils,
  Keras.PreProcessing,
  Windows;

var
x, y, x_test, y_test : TNDarray;

y_out2 : Variant;

xtestarray, ytestarray : TArray<Double>;

i : Longint;

begin

TNumPy.Init(True);

//Load train data

SetLength(xtestarray, 648);

for i := 0 to 648-1 do
  xtestarray[i] := Random;

x := TNumPy.npArray<Double>(xtestarray);
x := TNDArray(x.reshape([8, 9, 9, 1]));

SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY);
Writeln('x shape: '  + x.shape.ToString);
SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY);

y := TNumPy.npArray<Double>( [ 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0 ] );
y := TNDArray(y.reshape([8, 2]));


// --

    // input image dimensions
var img_rows: Integer := 9;
var img_cols: Integer := 9;

    // Declare the input shape for the network
var input_shape : Tnp_shape := default(Tnp_shape);

input_shape := Tnp_shape.Create([img_rows, img_cols, 1]);

    // Build CNN model
var model : TSequential := TSequential.Create;

model.Add( TConv2D.Create(32, [{3}3, {3}3],'relu', @input_shape) );
model.Add( TConv2D.Create(64, [{3}3, {3}3],'relu'));
model.Add( TMaxPooling2D.Create([{2}2, {2}2]));
model.Add( TDropout.Create({0.25}0.09));
model.Add( TFlatten.Create);
model.Add( TDense.Create(128, 'relu'));
model.Add( TDropout.Create({0.5}0.18));
model.Add( TDense.Create(2, 'softmax'));

model.Compile('adadelta', 'categorical_crossentropy',['accuracy']);


var batch_size : Integer := 128;
var history: THistory := model.Fit(x, y, @batch_size, 10000,1);

model.Summary;

var logs := history.HistoryLogs;


//Predict (network output)

SetLength(xtestarray, 648);

for i := 0 to 648-1 do
  xtestarray[i] := Random;

x_test := TNumPy.npArray<Double>(xtestarray);
x_test := TNDArray(x.reshape([8, 9, 9, 1]));


y_test := model.Predict(x_test);
y_test := TNDArray(y_test.reshape([16, 1]));
y_out2 := y_test.ToDoubleArray;

for i := 0 to 16-1 do
  Writeln('Network output on ', i, ' : ', y_out2[i]);

//Save model and weights
var json : string := model.ToJson;
TFile.WriteAllText('model.json', json);
model.SaveWeight('model.h5');

//Load model and weight
var loaded_model : TBaseModel := TSequential.ModelFromJson(TFile.ReadAllText('model.json'));
loaded_model.LoadWeight('model.h5');

Readln;

end.

//------------------------
