DatasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
ImpTimeSteps=[30,14,30,15,6,6, 40,40,20,20]
ImpFeatures=[30,14,30,15,40,40,6,6,20,20]

StartImpTimeSteps=[10,18,10,18,22,22,5,5,None,None ]
StartImpFeatures=[10,18,10,18,5,5,22,22,None,None ]

Loc1=[None,None,None,None,None,None,None,None,1,1]
Loc2=[None,None,None,None,None,None,None,None,29,29]


FreezeType = [None,None,None,None,None,None,None,None,"Feature","Time"]
isMoving=[False,False,True,True,False,True,False,True,None,None]
isPositional=[False,False,False,False,False,False,False,False,True,True]

DataGenerationTypes=[None ,"Harmonic"]

models=["Transformer" ,"LSTMWithInputCellAttention","TCN","LSTM"]

createDatasets(args,DatasetsTypes,ImpTimeSteps,ImpFeatures,StartImpTimeSteps,StartImpFeatures,Loc1,Loc2,FreezeType,isMoving,isPositional,DataGenerationTypes)