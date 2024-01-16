program test1;

uses
  Keras, Keras.Layers, Keras.Models,
  System.Types, np.Base, np.Api, np.Utils, np.Models, PythonEngine,
       Python.Utils, System.IOUtils, SysUtils;


var
  x, y, x_test, y_test : TNDarray;

  str_v : string;

  y_out : Variant;

begin


// ------------



TNumPy.Init(True);



//Load train data
//var x : TNDarray := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
//var y : TNDarray := TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );

x := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
y := TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );


//Build functional model
var input  : TKInput := TKInput.Create(tnp_shape.Create([2]));
var hidden1: TBaseLayer  := TDense.Create(32, 'relu').&Set([input]);
var hidden2: TBaseLayer  := TDense.Create(64, 'relu').&Set([hidden1]);
var output : TBaseLayer  := TDense.Create(1,  'sigmoid').&Set([hidden2]);


var model : TModel := TModel.Create ( [ input ] , [ output ]);



//Compile and train

model.Compile({TStringOrInstance.Create( TAdam.Create )}'adam', 'binary_crossentropy',['accuracy']);


var batch_size : Integer := 2;
var history: THistory := model.Fit(x, y, @batch_size, {10}500,1);

model.Summary;

var logs := history.HistoryLogs;

// predict (network output)

x_test := TNumPy.npArray<Double>( [ [ 0, 1 ] ] );

y_test := model.Predict(x_test);

str_v := model.Predict(x_test).ToString;

y_out := y_test.ToDoubleArray;

// predict (network output) - end

Writeln('Network output on [0,1]: ', y_out[0]);


//Save model and weights
var json : string := model.ToJson;
TFile.WriteAllText('model.json', json);
model.SaveWeight('model.h5');

//Load model and weight
var loaded_model : TBaseModel := TSequential.ModelFromJson(TFile.ReadAllText('model.json'));
loaded_model.LoadWeight('model.h5');

Readln;

// ------------


end.
