1. Install Delphi 10.3 Rio
2. Install Python 3.9 (64-bit)
3. Install TensorFlow 2.15.0
4. Download and unzip Keras4Delphi - debugged
5. Compile and run Project1.dproj or Project2.dproj in Delphi


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
