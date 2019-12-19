import pandas as pd
import numpy as np
import math
from scipy import sparse
from tqdm import tqdm
PATH = "D:/jobs/DL_practice/AIUP"

def process(x): 
    if "." in x:
        res = x.split(".")[0].zfill(2) 
    else:
        res = x
    res = int(res,16)
    return res

## for parallel 
def checkstat(md5,mode='train'):
    assert mode in ['train','test']
    tmp = pd.read_table(f"{PATH}/{mode}/{md5}.bytes", sep=' ',header=None,index_col=0)
    start = int(tmp.index[0][:-1],16)
    tmp = np.array(tmp).reshape([1,tmp.shape[0]*tmp.shape[1]])
    n = tmp.shape[1]
    end = start+n
    nums = {}
    for num in tmp.tolist()[0]:
        try:
            if len(num)>0:
                if num in nums:
                    nums[num] += 1
                else:
                    nums[num] = 1
        except:
            pass
    new_nums = {}
    for string in nums.keys():
        new_nums[process(string)] = nums[string]
    output = np.zeros(256)
    for i in range(256):
        if i in new_nums.keys():
            output[i] = new_nums[i]
        else:
            output[i] = 0
    output = output/n
    output = np.concatenate([output,[n,start,end]],axis=0)
    return output

def ngram(md5,mode,M1=1024,M2=100000,M =3):
    tmp = pd.read_table(f"{PATH}/{mode}/{md5}.bytes", sep=' ',header=None,index_col=0)
    start = int(tmp.index[0][:-1],16)
    tmp = np.array(tmp).reshape([1,tmp.shape[0]*tmp.shape[1]])
    tmp = tmp[~pd.isnull(tmp)].astype('str')
    end = start + len(tmp)
    T1,T2 = max(M1,start),min(M2,end)    
    tmp = np.array([process(x) for x in tmp[(T1-start):(T2-start) ] ])
    ans = np.zeros(M2)
    ans[T1:T2] = tmp
    ans = ans[M1:M2]
    cutsize = int((M2-M1)/M)
    result = np.zeros(cutsize,dtype='int')
    for i in range(cutsize):
        add_num = 0
        for idx,num in enumerate(ans[(i*M):(i*M+M)]):
            add_num +=num*( 10**( (M-idx-1)*M ) )
        result[i] = add_num
    return result

#from tqdm import tqdm
def ngram_stat(md5,mode='train',M=2):
    tmp = pd.read_table(f"{PATH}/{mode}/{md5}.bytes", sep=' ',header=None,index_col=0)
    start = int(tmp.index[0][:-1],16)
    tmp = np.array(tmp).reshape([1,tmp.shape[0]*tmp.shape[1]])
    tmp = tmp[~pd.isnull(tmp)].astype('str').astype('<U3')
    n = len(tmp)
    cutsize = int(n/M)
    nums = {}
    count0 = 0 
    for i in range(cutsize):
        for k in range(M):
            if '.' in tmp[i*M+k]:
                tmp[i*M+k] = tmp[i*M+k].split(".")[0].zfill(2) 
                count0+=1
            if len(tmp[i*M+k])==1:
                tmp[i*M+k] = '0'+tmp[i*M+k]
        if ''.join(tmp[(i*M):(i*M+M)]) in nums:
            nums[process(''.join(tmp[(i*M):(i*M+M)]))]+=1
        else:
            nums[process(''.join(tmp[(i*M):(i*M+M)]))]=1
    output = np.zeros(256**M)
    for i in nums.keys():
        output[i] = nums[i]
    output = output/n
    output = np.concatenate([output,[count0]],axis=0)
    return output

def Reduce2std(data,reduce_size):
    size1,size2 = data.shape
    output = np.zeros([size1,reduce_size])
    dcut = int(size2/reduce_size)
    for i in tqdm(range(reduce_size)):
        output[:,i] = data[:,i*dcut:i*dcut+dcut].toarray().std(axis=1)
    output = sparse.csr_matrix(output)
    return output

def GetDRdata(md5,M1=512,M2=18000000,mode='train',M=10000,DR_med='mean'):
    """ Get dimensional reduction redult.
    
        Parameters
        ---------------------------------------------------------
        M1     : int, md5 start index (512~4096)
        M2     : int, md5 end index   (10w~1900w)
        mode   : str, data type (train,test)
        M      : int, dimensional reduction size
        DR_med : str, dimensional reduction method (mean,max)
        ---------------------------------------------------------
    """
    assert mode in ['train','test']
    assert DR_med in ['mean','max','median','std']
    assert M2>M1
    assert M<=(M2-M1)
    tmp = pd.read_table(f"{PATH}/{mode}/{md5}.bytes", sep=' ',header=None,index_col=0)
    start = int(tmp.index[0][:-1],16)
    tmp = np.array(tmp).reshape([1,tmp.shape[0]*tmp.shape[1]])
    tmp = tmp[~pd.isnull(tmp)].astype('str')
    end = start + len(tmp)
    T1,T2 = max(M1,start),min(M2,end)    
    if T1>T2:
        return np.zeros(M)
    else:
        tmp = np.array([process(x) for x in tmp[(T1-start):(T2-start) ] ])
        ans = np.zeros(M2)
        ans[T1:T2] = tmp
        ans = ans[M1:M2]
        cutsize = int((M2-M1)/M)
        result = np.zeros(M)
        if M ==(M2-M1):
            return ans
        else:
            for i in range(M):
                if DR_med == 'mean':
                    result[i] = ans[(i*cutsize):(i*cutsize+cutsize)].mean()
                elif DR_med == 'max':
                    result[i] = ans[(i*cutsize):(i*cutsize+cutsize)].max()
                elif DR_med == 'median':
                    result[i] = np.median(ans[(i*cutsize):(i*cutsize+cutsize)])
                elif DR_med == 'std':
                    result[i] = ans[(i*cutsize):(i*cutsize+cutsize)].std()
            return result

