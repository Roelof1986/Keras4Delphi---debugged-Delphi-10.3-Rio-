program Project2;

uses
  Keras, Keras.Layers, Keras.Models,
  System.Types, np.Base, np.Api, np.Utils, np.Models, PythonEngine,
       Python.Utils, System.IOUtils, Vcl.PythonGUIInputOutput, Vcl.StdCtrls, Keras.PreProcessing, SysUtils, Windows;


var
  res      : TArray<TNDArray>;

//  redtOutput : TCustomMemo;

begin

TNumPy.Init(True);

    var max_features: Integer := 20000;
    // cut texts after this number of words (among top max_features most common words)
    var maxlen     : Integer := 80;
    var batch_size : Integer := 32;


//    redtOutput.Lines.Add('Loading data...');
//    TextColor(10);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    Writeln('Loading data...');
//    TextColor(7);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    res := TIMDB.load_data(@max_features);
    var x_train, y_train ,x_test, y_test,X,Y,tmp : TNDArray;

    x_train := res[0];
    y_train := res[1];
    x_test  := res[2];
    y_test  := res[3];

//    redtOutput.Lines.Add('train sequences: ' + x_train.shape.ToString);
//    redtOutput.Lines.Add('test sequences: '  + x_test.shape.ToString);

//    TextColor(10);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    Writeln('train sequences: ' + x_train.shape.ToString);
    Writeln('test sequences: '  + x_test.shape.ToString);
    Writeln('Pad sequences (samples x time)');
//    TextColor(7);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});

//    redtOutput.Lines.Add('Pad sequences (samples x time)');
    var tseq : TSequenceUtil := TSequenceUtil.Create;
    x_train := tseq.PadSequences(x_train, @maxlen);
    x_test  := tseq.PadSequences(x_test,  @maxlen);
//    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
//    redtOutput.Lines.Add('x_test shape: '  + x_test.shape.ToString);

//    TextColor(10);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    Writeln('x_train shape: ' + x_train.shape.ToString);
    Writeln('x_test shape: '  + x_test.shape.ToString);
    Writeln('Build model...');
//    TextColor(7);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});

//    redtOutput.Lines.Add('Build model...');
    var model : TSequential := TSequential.Create;
    model.Add( TEmbedding.Create(max_features, 128));
    model.Add( TLSTM.Create(128, 0.2, 0.2));
    model.Add( TDense.Create(1, 'sigmoid'));

    //try using different optimizers and different optimizer configs
//    model.Compile(TStringOrInstance.Create('adam'), 'binary_crossentropy', [ 'accuracy' ]);

    model.Compile({TStringOrInstance.Create( TAdam.Create )}'adam', 'binary_crossentropy',['accuracy']);
    model.Summary;

//    TextColor(10);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    Writeln('Train...');
//    TextColor(7);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});

//    redtOutput.Lines.Add('Train...');
    model.Fit(x_train, y_train, @batch_size, 15, 1,[ x_test, y_test ]);

    //Score the model for performance
    var score : TArray<Double> := model.Evaluate(x_test, y_test, @batch_size);

//    redtOutput.Lines.Add('Test score: '   + FloatToStr(score[0]));
//    redtOutput.Lines.Add('Test accuracy:'+ FloatToStr(score[1]));

//    TextColor(10);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});
    Writeln('Test score: '   + FloatToStr(score[0]));
    Writeln('Test accuracy:'+ FloatToStr(score[1]));
//    TextColor(7);
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED OR FOREGROUND_GREEN OR FOREGROUND_BLUE OR FOREGROUND_INTENSITY {OR BACKGROUND_GREEN});

    // Save the model to HDF5 format which can be loaded later or ported to other application
    model.Save('model.h5');

    Readln;


end.
