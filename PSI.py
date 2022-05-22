from _PSI import ffi,lib
DBG_FILE_NAME=ffi.new("char[]", 258)
a=[]
coefA=ffi.NULL
import sys
from io import StringIO

def ERROR_TEST2(iErr,msg):
    if iErr[0]!=0:
        print("Code d'erreur :"+str(iErr[0]))
        print(ffi.string(msg).decode("utf-8") )

        return ffi.string(msg).decode("utf-8")+"\n"
    return ""

class PSIDataClass():
    PSData = ffi.new("struct PSI_DATA_STRUCT *")
    iError = ffi.new("int *", 0)
    dFactor =1
    dBand=0
    arMsg = ffi.new("char[]", 258)
    arCoeffALocal= ffi.new("double[]", 32000)
    dDataStdEpsilon =  ffi.cast( 'double',10 )   #dFactor * 3. * (60 if dBand < 60 else dBand) / 1.73
    dCoeffAStdEpsilon = dFactor * 0.5
    a=100
    NbPtDetection=a
    NbPtCoeffADirect=a
    NbPtAAvg=a
    Ecart_min=10
    bOnlyFlatRangeAreTreated=1
    bSlopeIsUsedForFlatRange=0
    dCoeffAAvgEpsilon=0.1
    LeftTronc=0
    RightTronc=0



    def __init__(self,array):
        #buffer = StringIO()

        # redirect stdout to a buffer
        #sys.stdout = buffer
        self.Validated=""

        self.size=ffi.cast("long",len(array))

        self.tab=ffi.new("double[]",array)

        self.coefA=ffi.new("double[]",array)

        lib.DoPSInitialization(self.PSData,DBG_FILE_NAME,self.tab,
                               ffi.NULL,ffi.NULL,self.coefA,ffi.NULL,ffi.NULL,ffi.NULL,
                               self.size,self.NbPtDetection,self.NbPtCoeffADirect,
                               self.NbPtAAvg,self.bOnlyFlatRangeAreTreated,self.bSlopeIsUsedForFlatRange,
                               self.dDataStdEpsilon,self.dCoeffAStdEpsilon,self.dCoeffAAvgEpsilon,
                               self.Ecart_min,self.LeftTronc,self.RightTronc,self.iError,self.arMsg
        )
        self.Validated+=ERROR_TEST2(self.iError,self. arMsg)


        lib.DoPSCalculation(self.PSData,self.iError,self.arMsg)
        self.Validated+=ERROR_TEST2(self.iError,self. arMsg)

        lib.DoPSPostTreatmentOnSliceWidth( self.PSData,self.iError,self.arMsg )
        self.Validated+=ERROR_TEST2(self.iError,self. arMsg)

        lib.DoPSPostTreatmentOnSliceBounds(self.PSData, self.iError, self.arMsg)
        self.Validated+=ERROR_TEST2(self.iError,self. arMsg)

        #sys.stdout = sys.__stdout__

        print("bOnlyFlatRangeAreTreated:"+str(self.PSData.bOnlyFlatRangeAreTreated))
        print("bSlopeIsUsedForFlatRange: "+str(self.PSData.bSlopeIsUsedForFlatRange))
        print("dDataStdEpsilon:{0}\ndCoeffAStdEpsilon:{1}\ndCoeffAAvgEpsilon:{2}\nEcart_min:{3}\nNbPtDetection:{4}"
              .format(self.PSData.dDataStdEpsilon,self.PSData.dCoeffAStdEpsilon,self.PSData.dCoeffAAvgEpsilon,
                      self.PSData.dWidthMin,self.PSData.NbPtDetection))


    def getSliceList(self):
        return self.PSData.SliceList[0:self.PSData.nSliceCount]


if __name__ == "__main__":

    size=25000
    pente =False
    for i in range(size):
        if  1000>i:
            a.append(1000)
        elif 1000<=i<2000 and pente:
            a.append(1000+ 3000/1000 * (i-1000))
        elif 2000<=i<3000:
            a.append(4000)
        elif 3000 <= i < 4000:
            a.append( 1000)
        elif 4000<=i<5000:
            a.append(2000)
        elif 6000 <= i < 7000 and pente:
            a.append(2000/1000 * (i-6000))
        elif 7000<=i<8000:
            a.append(2000)
        elif 8000 <= i < 9000 and pente:
            a.append(2000-2000/1000 * (i-8000))
        elif 10000 <= i < 11000 and pente:
            a.append( 3000 / 1000 * (i - 10000))
        elif 11000<=i<12000 and pente:
            a.append( 3000-500/1000 * (i-11000))
        elif 12000<=i<13000 and pente:
            a.append(2500 - 2000/1000 *(i-12000))
        elif 13000<=i<14000:
            a.append(500)
        elif 14000 <= i < 22000 and pente:
            a.append(500+1500/8000 * (i-14000))

        else:
            a.append(0)

    psiDataClass=PSIDataClass(a)
    import matplotlib.pyplot as plt
    plt.plot(a)
    listA=psiDataClass.getSliceList()
    for slice in listA :
        a, b = slice.dLeftBound, slice. dRighBound
        plt.axline((a, 0), (a, .000001),color='blue',alpha=0.3)
        plt.axline((b, 0), (b, .000001),color='red',alpha=0.3)
        print("",a,b)
    print(str(len(listA))+" plateaux trouvés")

    plt.show()