def prob_analysis():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    sub='sub003'
    ls=[]
    probs=[]
    for curr_ses in range(2,5):
        # for currRun in range(1,11):
        currRun=1
        while True:
            try:
                history=pd.read_csv(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/{sub}/ses{curr_ses}/feedback/{sub}_{currRun}_history.csv")
                l = list(history[history['states']=="feedback"]['B_prob'])
                ls.append(np.mean(l))
                # print(np.mean(l))
                if len(probs)==0:
                    probs = np.expand_dims(l,0)
                else:
                    probs = np.concatenate([probs,np.expand_dims(l,0)],axis=0) 
                currRun+=1
            except:
                break
    _=plt.figure()
    plt.plot(ls)
    plt.xlabel("run ID")
    plt.ylabel("mean prob of only feedback TRs in that run")
    plt.title("mean prob of only feedback TRs in that run v.s. run ID")
    print(f"mean prob of all feedback TRs={np.mean(ls)}")





    import seaborn as sns
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")
    # ax = sns.boxplot(x=probs)
    # np.expand_dims(l,0).shape
    # probs.shape
    # sns.boxplot(probs)
    _=plt.figure(figsize=(20,20))
    _=plt.boxplot(probs.T)
    for currRun in range(len(probs)):
        plt.scatter([currRun+1+0.1]*60,probs[currRun],s=1)
    _=plt.xlabel("run ID")
    _=plt.ylabel("prob")
    _=plt.plot(np.arange(1,len(probs)+1),ls)





    import pandas as pd
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats

    y = probs.reshape(np.prod(probs.shape))
    X=[]
    for currRun in range(1,len(probs)+1):
        X+=[currRun]*60

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    print("Ordinary least squares")



    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def gaussian_fit(y=[1,2,3,4],x=1):
        y_values = np.linspace(min(y), max(y), 120)    
        mu=np.mean(y)
        sig=np.std(y)
        #plt.text(x+0.04, mu, 'mu={0:.2f}\nsig={1:.2f}'.format(mu,sig), fontsize=12)
        plt.plot(x+0.04+0.5*gaussian(y_values, mu, sig),y_values)
        

    _=plt.figure(figsize=(10,10))
    for currRun in range(len(probs)):
        plt.scatter([currRun]*60,probs[currRun],s=1)
        gaussian_fit(y=probs[currRun],x=currRun)
    b=0.0061
    const=0.3880

    plt.plot(np.arange(len(probs)),np.arange(len(probs))*b+const)
    plt.xlabel("run ID")
    plt.ylabel("probability")
    plt.title("Gaussian fitted probability distribution")



# Trained axis v.s. Untrained axis
# Drift happening 
def plotForceGreedyAccCurve():
    way4 = [0.7401374113475178,0.6668513593380615,0.653125,0.5723168169904409,0.7005208333333334]
    way2 = [0.8607714371980677,0.8386754776021079,0.8017882630654369,0.7814661561264822,0.8541666666666666]
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    axs[0].plot(np.arange(1,len(way4)+1),new_trained_full_rotation_4_way_accuracy_mean,label="4_way new mask",color="orange")
    axs[0].plot(np.arange(1,len(way4)+1),mean_of_2_way_clf_acc_full_rotation,label="2_way new mask",color="red")
    axs[0].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
    axs[0].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
    axs[0].legend()
    axs[0].set_ylabel("acc")
    axs[0].set_xlabel("session ID")
    axs[0].set_ylim([0.24,0.9])
    axs[0].set_title("using new mask")


    way4=[0.7401374113475178,0.5740543735224587,0.5038194444444444,0.4748424491211841,0.546875]
    way2=[0.8607714371980677,0.8114603919631094,0.7822690217391304,0.7463665184453228,0.8159722222222222]
    axs[1].plot(np.arange(1,len(way4)+1),way4,label="4_way old mask",color="orange")
    axs[1].plot(np.arange(1,len(way4)+1),way2,label="2_way old mask",color="red")
    axs[1].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
    axs[1].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
    axs[1].legend()
    axs[1].set_ylabel("acc")
    axs[1].set_xlabel("session ID")
    axs[1].set_ylim([0.24,0.9])
    axs[1].set_title("using ses1 mask")


def megaROIOverlapping():
    import os
    import re
    from glob import glob



    os.chdir("/gpfs/milgram/pi/turk-browne/projects/rtSynth_rt/")
    def findDir(path):
        from glob import glob
        # _path = glob(path)[0]+'/'
        _path = glob(path)
        if len(_path)==0: # if the dir is not found. get rid of the "*" and return
            _path=path.split("*")
            _path=''.join(_path)
        else:
            _path = _path[0]+'/'
        return _path

    def getBestROIs(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        bestROIs = txt.split("bestROIs=(")[1].split(")\n/gpfs/milgram/")[0]
        bestROIs = bestROIs.split("', '")
        bestROIs = [re.findall(r'\d+', i)[0] for i in bestROIs]

        return bestROIs
    def get4wayacc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("new trained full rotation 4 way accuracy mean=")[1].split("\nnew_run_indexs")[0])
        return acc
    def get2wayacc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("mean of 2 way clf acc full rotation =")[1].split("\nbedbench_bedchair")[0])
        return acc
    def getAB_acc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("mean of 2 way clf acc full rotation =")[1].split("\nbedbench_bedchair")[0])
        return acc

    logID={1:'00000181',2:'17800181',3:'17800357',4:'17806127',5:'17800522'}
    BestROIs=[]
    FourWayAcc=[]
    TwoWayAcc=[]
    AB_acc=[]
    for currSess in range(1,6):
        BestROIs.append(getBestROIs(logID=logID[currSess]))
        FourWayAcc.append(get4wayacc(logID=logID[currSess]))
        TwoWayAcc.append(get2wayacc(logID=logID[currSess]))
        AB_acc.append(getAB_acc(logID=logID[currSess]))
    BestROIs

    _bestROIs=[]
    for i in BestROIs:
        _bestROIs.append([int(a) for a in i])

    _=plt.figure(figsize=(10,10))
    for currSession in range(1,len(_bestROIs)+1):
        plt.scatter(_bestROIs[currSession-1],[currSession]*len(_bestROIs[currSession-1]),s=10,label=f"session{currSession}")
    plt.title("compare the best ROI selected from each session")    
    plt.ylabel("session")
    plt.xlabel("ROI ID")





    def inRatio(a1,a2):
        count=0
        for i in a2:
            if i in a1:
                count+=1
        return count/len(a1)
    plt.figure(figsize=(16, 8)) 
    for i in range(1,6):
        ratios=[]
        for currSession in range(1,6):
            t = inRatio(_bestROIs[i-1],_bestROIs[currSession-1])
            ratios.append(t)
        plt.subplot(1,6,i)
        plt.plot(np.arange(1,6),ratios)
        plt.xlabel("session")
        plt.ylim([0,1])
        plt.title(f"ROI in ses x out of ses{i}")



    array=np.zeros((5,300))
    for currROI in range(1,301):
        for currSes in range(1,6):
            if currROI in _bestROIs[currSes-1]:
                array[currSes-1, currROI-1] = 1
    # plt.imshow(array)
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(1,301),np.sum(array,axis=0))
    plt.xlabel("ROI ID")
    plt.ylabel("count of existence")



    arrayMean=np.sum(array,axis=0)
    print(f"ID of survived ROI in all sessions={np.where(arrayMean==5)[0]+1},possible range 1-300")

    arrayMean=np.sum(array,axis=0)
    arrayMean[arrayMean==0]=None
    plt.hist(arrayMean)
    #plt.xlim([0.9,6.5])




    # cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
    # load_fsl
    # fslview_deprecated GMschaefer_8.nii.gz GMschaefer_159.nii.gz GMschaefer_160.nii.gz GMschaefer_163.nii.gz
    # 8 159 160 163


    def GreedySum(bestROIs=None,sub=None):
        import nibabel as nib
        import pandas as pd
        workingDir=f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/{sub}/ses1/recognition/mask/"
        for pn, parc in enumerate(bestROIs):
            _mask = nib.load(workingDir+f"GMschaefer_{parc}")
            aff = _mask.affine
            _mask = _mask.get_data()
            _mask = _mask.astype(int)
            mask = _mask if pn == 0 else mask + _mask
            
        savemask = nib.Nifti1Image(mask, affine=aff)
        nib.save(savemask, f"{workingDir}GreedySum.nii.gz")
        
    for i in _bestROIs:
        for j in i:
            print(f"{j}.nii.gz', '",end='')
            
    bestROIs_allSes = ('8.nii.gz', '159.nii.gz', '235.nii.gz', '163.nii.gz', '271.nii.gz', '164.nii.gz', '258.nii.gz', '80.nii.gz', '218.nii.gz', '67.nii.gz', '211.nii.gz', '2.nii.gz', '220.nii.gz', '62.nii.gz', '160.nii.gz', '22.nii.gz', '79.nii.gz', '8.nii.gz', '159.nii.gz', '114.nii.gz', '235.nii.gz', '163.nii.gz', '164.nii.gz', '151.nii.gz', '80.nii.gz', '112.nii.gz', '126.nii.gz', '67.nii.gz', '209.nii.gz', '211.nii.gz', '205.nii.gz', '2.nii.gz', '39.nii.gz', '160.nii.gz', '244.nii.gz', '8.nii.gz', '159.nii.gz', '195.nii.gz', '163.nii.gz', '164.nii.gz', '151.nii.gz', '80.nii.gz', '58.nii.gz', '67.nii.gz', '209.nii.gz', '211.nii.gz', '150.nii.gz', '160.nii.gz', '246.nii.gz', '8.nii.gz', '223.nii.gz', '159.nii.gz', '163.nii.gz', '271.nii.gz', '80.nii.gz', '126.nii.gz', '67.nii.gz', '146.nii.gz', '2.nii.gz', '160.nii.gz', '246.nii.gz', '22.nii.gz', '244.nii.gz', '30.nii.gz', '8.nii.gz', '86.nii.gz', '159.nii.gz', '195.nii.gz', '163.nii.gz', '56.nii.gz', '77.nii.gz', '76.nii.gz', '263.nii.gz', '62.nii.gz', '281.nii.gz', '160.nii.gz', '30.nii.gz', '79.nii.gz')
    GreedySum(bestROIs=bestROIs_allSes,sub='sub003')
    
    # cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
    # fslview_deprecated GreedySum.nii.gz ../WANGinFUNC.nii.gz templateFunctionalVolume_bet.nii.gz ../templateFunctionalVolume_bet.nii.gz
