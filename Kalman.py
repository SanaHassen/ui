

from _Code_KALMAN import ffi,lib
#Kalman variable :
TYPE_DROIT=0
TYPE_SINUS=1

def ERROR_TEST2(iErr,msg):
    if iErr[0]!=0:
        print("Code d'erreur :"+str(iErr[0]))
        print(ffi.string(msg))

def FiltreKalmanInitialisation(Tdata ,DBG_FILE_NAME = ffi.new("char[]", 258),VBandW0 = 3.465,
    VBandW1 = 3.465,QRef0 = 0.001,QRef1 = 0.0001 ,QFactor0 = 1,QFactor1 = 1,RRef0 = 1000,RRef1 = 0.1,RFactor0 = 1,RFactor1 = 1,
    X_real0_0 = 10,X_real0_1 = 5,RNoise0=ffi.new("double *", 1),RNoise1 = ffi.new("double *", 1),Q0 = ffi.new("double *", 1),
    Q1 = ffi.new("double *", 1),R0 = ffi.new("double *", 1),R1 = ffi.new("double *", 1),iError = ffi.new("int *", 0),
    arMsg = ffi.new("char[]", 258),
    KFData = ffi.new("struct KFI_DATA_STRUCT *"),NStatPt = 25,dTimeStep = 1,forme:int=TYPE_DROIT,Frequence = 1,P0_00 = 1,P0_11 = 1,
                               X_est0_0 = 10,X_est0_1 = 0):

    lib.DoTInitialization(Tdata, DBG_FILE_NAME, VBandW0, VBandW1, QRef0, QRef1, QFactor0, QFactor1, RRef0, RRef1,
                          RFactor0, RFactor1, X_real0_0,
                          X_real0_1, RNoise0, RNoise1, Q0, Q1, R0, R1, iError, arMsg)

    lib.DoKFInitialization(KFData, DBG_FILE_NAME, NStatPt, dTimeStep,forme ,Frequence, Q0[0], Q1[0], R0[0], R1[0], P0_00,
                           P0_11, X_est0_0, X_est0_1, iError, arMsg)

#renvoie le point caculé
def FiltreKalmanOneStep(KFData,Value,iError, arMsg,MType=lib.TYPE_DROIT,dTimeStep=1) :
    '''
    # définition de la Matrice B
    lib.SetMatrixToNull(KFData.mtB, lib.X_SIZE, lib.Y_SIZE, iError, arMsg)
    ERROR_TEST2(iError, arMsg)
    KFData.vxG[lib.POSITION] = 0
    KFData.vxG[lib.VELOCITY] = 0
    '''
    KFData.vxY_mes[lib.POSITION] = Value

    lib.DoKFOneStep(KFData, iError, arMsg)
    ERROR_TEST2(iError, arMsg)

    # Récupération des valeurs
    x0_est = ffi.new("double *")
    x1_est = ffi.new("double *")
    lib.GetVectorValue(KFData.vxX_est, x0_est, x1_est, iError, arMsg)
    ERROR_TEST2(iError, arMsg)

    DerMes = KFData.DerMes
    DerEst = KFData.DerEst

    VEstAvg = KFData.dVEstAvg
    VEstStd = KFData.dVEstStd
    VEstStdNV = 100*VEstStd/(1.E-07+VEstAvg)

    return x1_est[0],x0_est[0],DerMes,DerEst,VEstAvg,VEstStd,VEstStdNV

#Renvoie  ListQ,listM,ListDMes,ListDEst
def GetKalmanResults(ListMasse,Tdata = ffi.new("struct KFI_TEST_STRUCT *"),KFData = ffi.new("struct KFI_DATA_STRUCT *"),
    iError = ffi.new("int *", 0),arMsg = ffi.new("char[]", 258),dTimeStep=1,X_real0_1=0,QRef0 = 0.001,QRef1 = 0.0001,
                 RRef0 = 1000,RRef1 = 0.1,freq=1,forme=TYPE_DROIT):

    X_real0_0=ListMasse[0]
    FiltreKalmanInitialisation(Tdata=Tdata, KFData=KFData, iError=iError, arMsg=arMsg,
                               X_real0_0=X_real0_0, X_real0_1=X_real0_1,X_est0_0=X_real0_0,
                               QRef0=QRef0,QRef1=QRef1,RRef0=RRef0,RRef1=RRef1,forme=forme,Frequence=freq)

    ListQ=[]
    ListM=[]
    ListDMes=[]
    ListDEst=[]
    ListVEstAvg=[]
    ListVEstStd=[]
    ListVEstStdNV=[]
    for i in range(len(ListMasse)):
        a,b,DMes,DEst,VEstAvg,VEstStd,VEstStdNV = FiltreKalmanOneStep(KFData, ListMasse[i], iError, arMsg)

        ListQ.append(a*3600/dTimeStep)
        ListM.append(b)
        ListDMes.append(DMes)
        ListDEst.append(DEst)
        ListVEstAvg.append(VEstAvg)
        ListVEstStd.append(VEstStd)
        ListVEstStdNV.append(VEstStdNV)


    return ListQ, ListM,ListDMes,\
           ListDEst, ListVEstAvg, ListVEstStd, ListVEstStdNV

