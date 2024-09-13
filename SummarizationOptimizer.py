import pandas as pd
import numpy as np
import re
import nltk
import emoji
import math
import operator
import copy

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer

import markov_clustering as mc
import networkx as nx
import random

import requests

from BatAlgorithm import *

from pso import pso_simple
from pso.cost_functions import sphere

from ecabc import ABC

#Penyimpanan Fungsi Preprocessing Text
def inject_preprocessing(obj_preprocess):
  global PreProcessDataTrain
  PreProcessDataTrain = obj_preprocess

class Preprocessing:
    def __init__(self):
        """Dekalrasi Class Untuk Stemming"""
        self.factory   = StemmerFactory()
        self.stemmer   = self.factory.create_stemmer()
        self.stopWords = self.getStopWordList()

    def give_emoji_free_text(self,text):
        """Menghilangkan Emoji Pada Tweet"""
        emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
        return clean_text

    def url_free_text(self,text):
        """Menghilangkan Url Tweet"""
        text = re.sub(r'http\S+', '', text)
        return text

    def username_free_text(self,text):
        """Menghilangkan Username User"""
        result = re.sub(r'@\S+','', text)
        v = re.sub(r"^b'RT|^b'",'', result)
        return v

    def replaceTwoOrMore(self,s):
        """Menghilangkan Karakter Berulang"""
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)

    def getStopWordList(self):
        """Mengambil Stopword dari Library Sastrawi"""
        stopWords = StopWordRemoverFactory()
        more_stopword = ['dengan', 'ia','bahwa','oleh','AT_USER','URL','di','yg','dari','ke','ini','bgmn','tmn2','dr','pt','dg','prn','bn','sbb', 'tdk', 'krn', 'ga,',
                        'tak', 'gak', 'gk', 'bkn', 'kan', 'la', 'so', 'dgn', ]
        data = stopWords.get_stop_words()+more_stopword

        return data

    def steaming_text(self,sentence):
        """Stemming Pada Text"""
        return self.stemmer.stem(sentence)

    def getFeatureVector(self,tweet):
        """Melakukan Tokenisasi"""
        featureVector = []
        words = tweet.split()
        for w in words:
            w = self.replaceTwoOrMore(w)
            w = w.strip('\'"?,.').lower()
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            if(w in self.stopWords or val is None):
                continue
            else:
                featureVector.append(w)
        return featureVector

#Preprocessing Text
class PreprocessingText(Preprocessing):
    def __init__(self,dataset):
        Preprocessing.__init__(self)
        """Data yang di Inputkan merupakan nama file csv tempat tweet disimpan"""
        self.csvFile = dataset
        self.processingTweet()

    def processingTweet(self):
        call_emoji_free = lambda x: self.give_emoji_free_text(x)
        self.csvFile['cleanTeks'] = self.csvFile['teks'].apply(call_emoji_free)
        self.csvFile['cleanTeks'] = self.csvFile['cleanTeks'].apply(self.url_free_text)
        self.csvFile['cleanTeks'] = self.csvFile['cleanTeks'].apply(self.username_free_text)
        self.csvFile['cleanTeks'] = self.csvFile['cleanTeks'].apply(self.getFeatureVector)#Hasil stopword removal
        self.csvFile['cleanTeks'] = [' '.join(map(str, l)) for l in self.csvFile['cleanTeks']]
        self.csvFile['cleanTeks'] = self.csvFile['cleanTeks'].apply(self.steaming_text)#Steaming

        self.csvFile['cleanTitle'] = self.csvFile['title'].apply(call_emoji_free)
        self.csvFile['cleanTitle'] = self.csvFile['cleanTitle'].apply(self.url_free_text)
        self.csvFile['cleanTitle'] = self.csvFile['cleanTitle'].apply(self.username_free_text)
        self.csvFile['cleanTitle'] = self.csvFile['cleanTitle'].apply(self.getFeatureVector)#Hasil stopword removal
        self.csvFile['cleanTitle'] = [' '.join(map(str, l)) for l in self.csvFile['cleanTitle']]
        self.csvFile['cleanTitle'] = self.csvFile['cleanTitle'].apply(self.steaming_text)#Steaming
        self.csvFile = self.csvFile.dropna()
        self.csvFile = self.csvFile[self.csvFile.cleanTeks != '']

#Data Preparation
class Preparation_Data(PreprocessingText):
  def __init__(self,df):
    df_split = pd.DataFrame.from_dict(self.teks_extract(df))
    self.df_clean = PreprocessingText(df_split).csvFile.reset_index(drop=True)

  def dotImportant(self,text):
    text = re.sub(r'(Rp [0-9]*(.)[0-9]*)', lambda x:x.group(0).replace(".",''), text)
    text = re.sub(r'([0-9a-zA-Z]*.com)', lambda x:x.group(0).replace(".",''), text)
    return text

  def teks_extract(self,df):
    df_split = {
        'title' : [],
        'teks' : [],
        'tanggal' : [],
        'doc_id' : [],
        'teks_id' : []
    }
    for i in range(len(df)):
      for j,text in enumerate(self.dotImportant(df.iloc[[i]]['content'].values[0]).split('. ')):
        df_split['title'].append(df.iloc[[i]]['title'].values[0])
        df_split['tanggal'].append(df.iloc[[i]]['Tanggal'].values[0])
        df_split['teks'].append(text.strip())
        df_split['doc_id'].append(i)
        df_split['teks_id'].append(j)

    return df_split

#Feature Ekstraction
class FeatureExtraction(Preparation_Data):
  def __init__(self,df,tfidf=True,tfidf_compress="",cleaning = True):
    self.vectorizer = TfidfVectorizer()
    self.tfidf_compress = tfidf_compress
    self.X_data = None

    df_clean = None
    if cleaning:
      df_clean = Preparation_Data(df).df_clean
    else:
      df_clean = df
      print("Data Bersih Masuk Sayang")

    self.df_clean = df_clean

    self.fitur1 = df_clean.copy()
    self.fitur2 = df_clean.copy()
    self.fitur3 = df_clean.copy()
    self.fitur4 = df_clean.copy()
    self.fitur5 = df_clean.copy()
    self.fitur6 = df_clean.copy()
    self.fitur7 = df_clean.copy()
    self.fitur8 = df_clean.copy()
    self.fitur9 = df_clean.copy()

    df_clean = self.Fitur1(df_clean)
    print("sukses Fitur 1")
    df_clean = self.Fitur2(df_clean)
    print("sukses Fitur 2")
    df_clean = self.Fitur3(df_clean)
    print("sukses Fitur 3")
    df_clean = self.Fitur4(df_clean)
    print("sukses Fitur 4")
    df_clean = self.Fitur5(df_clean)
    print("sukses Fitur 5")
    df_clean = self.Fitur6(df_clean)
    print("sukses Fitur 6")
    df_clean = self.Fitur7(df_clean)
    print("sukses Fitur 7")

    if tfidf:
      self.df_fiks = self.Fitur8(df_clean)
      print("sukses Fitur 8")

  def transform(self,df):
    df_clean = Preparation_Data(df).df_clean
    self.df_clean = df_clean
    self.fitur1 = df_clean.copy()
    self.fitur2 = df_clean.copy()
    self.fitur3 = df_clean.copy()
    self.fitur4 = df_clean.copy()
    self.fitur5 = df_clean.copy()
    self.fitur6 = df_clean.copy()
    self.fitur7 = df_clean.copy()
    self.fitur8 = df_clean.copy()
    self.fitur9 = df_clean.copy()

    df_clean = self.Fitur1(df_clean)
    df_clean = self.Fitur2(df_clean)
    df_clean = self.Fitur3(df_clean)
    df_clean = self.Fitur4(df_clean)
    df_clean = self.Fitur5(df_clean)
    df_clean = self.Fitur6(df_clean)
    df_clean = self.Fitur7(df_clean)
    return self.Fitur8_fit(df_clean)

  def KemiripanJudul(self,judul,kalimat):
    #Deklarasi Variabel Untuk menyimpan Jumlah kata yang sama dengan judul dan jumlah kata dalam judul
    jumlah_kata_sama_dengan_judul = 0
    jumlah_kata_dalam_judul       = len(judul)

    #Mengecek
    for j in judul:
      """Pengecekan similarity menggunakan distance antar kata dengan
      membandingkan 2 kata dan nltk.edit_distance akan mencari jumlah huruf yang berbeda.
      Bila jumlah huruf yang berbeda kurang dari setengah panjang huruf kata dengan similarity terkecil
      maka jumlah_kata_sama_dengan_judul akan bertambah 1"""
      indexList = [nltk.edit_distance(j, x) for x in kalimat]
      if min(indexList)<int(len(kalimat[indexList.index(min(indexList))])*0.5):
        jumlah_kata_sama_dengan_judul+=1
    return jumlah_kata_sama_dengan_judul/jumlah_kata_dalam_judul

  def Fitur1(self,df_clean):
    fitur1 = []
    for i in range(len(df_clean)):
      kalimat = df_clean.iloc[[i]].cleanTeks.values[0].split()
      judul = df_clean.iloc[[i]].cleanTitle.values[0].split()
      fitur1.append(self.KemiripanJudul(judul,kalimat))
    df_clean['fitur1'] = np.array(fitur1)
    self.fitur1['fitur1'] = np.array(fitur1)

    return df_clean

  def Fitur2(self,df_clean):
    fitur2 = []
    for i in range(len(df_clean)):
      kalimat = len(df_clean.iloc[[i]].cleanTeks.values[0].split())
      kalimatmax = max([len(i.split()) for i in df_clean[df_clean['doc_id']==df_clean.iloc[[i]].doc_id.values[0]].cleanTeks.values.tolist()])
      fitur2.append(kalimat/kalimatmax)
    df_clean['fitur2'] = np.array(fitur2)
    self.fitur2['fitur2'] = np.array(fitur2)

    return df_clean

  def Fitur3(self,df_clean):
    fitur3 = []
    for i in range(len(df_clean)):
      kalimat = len(df_clean[df_clean['doc_id']==df_clean.iloc[[i]].doc_id.values[0]])
      num = kalimat-df_clean.iloc[[i]].teks_id.values[0]
      fitur3.append(num/kalimat)
    df_clean['fitur3'] = np.array(fitur3)
    self.fitur3['fitur3'] = np.array(fitur3)

    return df_clean

  def Fitur4(self,df_clean):
    fitur4 = []
    for i in range(len(df_clean)):
      try:
        document = df_clean[df_clean['doc_id']==df_clean.iloc[[i]].doc_id.values[0]].cleanTeks.values.tolist()
        j = df_clean.iloc[[i]].cleanTeks.values[0]
        cosins = []
        for k in document:
          if j!=k:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([j, k])
            similarity = cosine_similarity(vectors[0], vectors[1])
            cosins.append(similarity[0][0])

        hasil = sum(cosins)/max(cosins)
        if np.isnan(hasil):
          fitur4.append(0)
        else:
          fitur4.append(hasil)
      except:
        fitur4.append(0)
    df_clean['fitur4'] = np.array(fitur4)
    self.fitur4['fitur4'] = np.array(fitur4)

    return df_clean

  def Fitur5(self,df_clean):
    my_file = open("Indonesian-Word-Tagged/resources/word-noun.txt", "r")
    noun = my_file.read().split("\n")
    fitur5 = []
    for i in range(len(df_clean)):
      kal = df_clean.iloc[[i]].cleanTeks.values[0]
      num = 0
      for j in kal.split(" "):
        for n in noun:
          if n==j:
            num+=1

      fitur5.append(num/len(kal.split(" ")))
    df_clean['fitur5'] = np.array(fitur5)
    self.fitur5['fitur5'] = np.array(fitur5)

    return df_clean

  def Fitur6(self,df_clean):
    fitur6 = []
    for i in df_clean.doc_id.unique():
      document = df_clean[df_clean['doc_id']==i].cleanTeks.values.tolist()
      doc_rank = {}
      for kal in document:
        for kata in kal.split(' '):
          if kata not in doc_rank.keys():
            doc_rank[kata] = 0
          doc_rank[kata]+=1
      doc      = dict(sorted(doc_rank.items(), key=operator.itemgetter(1),reverse=True))
      doc_rank = list(doc.keys())
      num_word = list(doc.values())[0]
      for kal in document:
        num = 0
        for kata in kal.split(' '):
          for dr in doc_rank[:10]:
            if kata==dr:
              num+=1
        fitur6.append(num/num_word)
    df_clean['fitur6'] = np.array(fitur6)
    self.fitur6['fitur6'] = np.array(fitur6)

    return df_clean

  def Fitur7(self,df_clean):
    my_file = open("pujangga/resource/netagger/morphologicalfeature/number.txt", "r")
    number = my_file.read().split("\n") + ['pertama']

    fitur7 = []
    for i in range(len(df_clean)):
      kal = df_clean.iloc[[i]].cleanTeks.values[0]
      num = 0
      for j in kal.split(" "):
        if j in number:
          num+=1
      fitur7.append(num/len(kal.split(" ")))
    df_clean['fitur7'] = np.array(fitur7)
    self.fitur7['fitur7'] = np.array(fitur7)

    return df_clean

  def Fitur8(self,df_clean):
    corpus = df_clean.cleanTeks.tolist()
    X = self.vectorizer.fit_transform(corpus)

    if self.tfidf_compress=='sum':
      sum = X.sum(axis=1)
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = sum,columns = ['fitur8'])],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = sum,columns = ['fitur8'])],axis=1)
    elif self.tfidf_compress=='mean':
      mean = X.mean(axis=1)
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = mean,columns = ['fitur8'])],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = mean,columns = ['fitur8'])],axis=1)
    else:
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = X.toarray(),columns = self.vectorizer.get_feature_names_out())],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = X.toarray(),columns = self.vectorizer.get_feature_names_out())],axis=1)

    return df_fiks

  def Fitur8_fit(self,df_clean):
    corpus = df_clean.cleanTeks.tolist()
    X = self.vectorizer.transform(corpus)

    if self.tfidf_compress=='sum':
      sum = X.sum(axis=1)
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = sum,columns = ['fitur8'])],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = sum,columns = ['fitur8'])],axis=1)
    elif self.tfidf_compress=='mean':
      mean = X.mean(axis=1)
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = mean,columns = ['fitur8'])],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = mean,columns = ['fitur8'])],axis=1)
    else:
      df_fiks = pd.concat([df_clean,pd.DataFrame(data = X.toarray(),columns = self.vectorizer.get_feature_names_out())],axis=1)
      self.fitur8 = pd.concat([self.fitur8,pd.DataFrame(data = X.toarray(),columns = self.vectorizer.get_feature_names_out())],axis=1)

    return df_fiks

#Preprocessing Tuna
class PreprocessTuna(FeatureExtraction):
  def __init__(self,df,sin_conv,treshhold,markov,tfidf=True,tfidf_compress="",extrackFitur=None):
    if extrackFitur is not None:
      print("success Open Data")
      self.extrackFitur = extrackFitur
    else:
      self.extrackFitur = FeatureExtraction(df,tfidf,tfidf_compress)
      
    try:
      self.fitur1 = self.extrackFitur.fitur1
    except:
      self.fitur1 = self.extrackFitur.df_clean

    try:
      self.fitur2 = self.extrackFitur.fitur2
    except:
      self.fitur2 = self.extrackFitur.df_clean
    
    try:
      self.fitur3 = self.extrackFitur.fitur3
    except:
      self.fitur3 = self.extrackFitur.df_clean

    try:
      self.fitur4 = self.extrackFitur.fitur4
    except:
      self.fitur4 = self.extrackFitur.df_clean

    try:  
      self.fitur5 = self.extrackFitur.fitur5
    except:
      self.fitur5 = self.extrackFitur.df_clean

    try:  
      self.fitur6 = self.extrackFitur.fitur6
    except:
      self.fitur6 = self.extrackFitur.df_clean

    try:  
      self.fitur7 = self.extrackFitur.fitur7
    except:
      self.fitur7 = self.extrackFitur.df_clean

    try:  
      self.fitur8 = self.extrackFitur.fitur8
    except:
      self.fitur8 = self.extrackFitur.df_clean

    try:  
      self.fitur9 = self.extrackFitur.fitur9
    except:
      self.fitur9 = self.extrackFitur.df_clean

    self.sin_conv = sin_conv
    self.markov = markov
    print("success Fitur Copy")

    self.df_fiks = None
    # if treshhold:
    #   #Sentence elimination
    #   self.df_fiks = self.eliminasi_treshhold(self.extrackFitur.df_fiks)
    # else:
    #   self.df_fiks = self.extrackFitur.df_fiks

    self.df_fiks = self.eliminasi_treshhold(self.extrackFitur.df_fiks.replace({True: 1, False: 0})).fillna(0)
    print("success on elimination")

    g,s = self.grafCreate(self.df_fiks,'cleanTeks')
    print("success Graft Create")

    self.df_graf = pd.DataFrame(g)
    self.df_fiks = pd.concat([self.df_fiks,pd.DataFrame.from_dict(s)],axis=1)

    #Cluster 1
    clusters = self.get_best_inflitation(self.df_graf)
    self.df_fiks['Cluster1'] = None
    for i in range(len(clusters)):
      for j in range(len(clusters[i])):
        self.df_fiks['Cluster1'][clusters[i][j]] = i

    NC = len(clusters)
    print("success Clustering 1")

    #Sinonim Conversion
    if sin_conv:
      resp = requests.get('https://raw.githubusercontent.com/victoriasovereigne/tesaurus/master/dict.json')
      resp_dict = resp.json()
      Sinonim1 = []
      for j in range(len(self.df_fiks)):
        kal = self.df_fiks.iloc[[j]].cleanTeks.values[0]
        newKal = []
        for k in kal.split(" "):
          try:
            k = resp_dict[k]['sinonim'][0]
          except:
            pass
          newKal.append(k)
        Sinonim1.append(" ".join(newKal))
      #Hasil klaster kalimat (Pertama)
      self.df_fiks['Sinonim1'] = np.array(Sinonim1)
    else:
      self.df_fiks['Sinonim1'] = self.df_fiks.cleanTeks
    
    print("success Sinonim Process")

    #Cluster 2
    self.df_fiks2 = None
    self.df_to_TSO = None
    if markov:
      for clus in range(NC):
        try:
          df_reclus = self.df_fiks[self.df_fiks['Cluster1']==clus].copy()
          g,s = self.grafCreate(df_reclus,'Sinonim1')
          df_graf = pd.DataFrame(g)
          df_reclus['Cluster2'] = 0

          #Pengembangan Selanjutnya Untuk Pengujian Treshold
          df_reclus['TresholdMCL'] = df_graf.mean().values < df_graf.mean().quantile(0.75)

          clusters = self.get_best_inflitation(df_graf)
          for i in range(len(clusters)):
            for j in range(len(clusters[i])):
              df_reclus['Cluster2'][clusters[i][j]] = i
        except:
          df_reclus['Cluster2']=0
        #Hasil klaster kalimat (Kedua)
        self.df_fiks2 = pd.concat([self.df_fiks2,df_reclus])
      #Hasil akhir penanganan Redundasi
      #self.df_to_TSO = self.df_fiks2[self.df_fiks2['TresholdMCL']==True].reset_index(drop=True).fillna(0)
      self.df_to_TSO = self.df_fiks2.fillna(0)
    else:
      #Hasil akhir penanganan Redundasi
      self.df_to_TSO = self.df_fiks.fillna(0)

    print("success Clustering 2")

  def eliminasi_treshhold(self,df_fiks):
    isDrop = []
    number_form = []
    treshhold    = df_fiks.iloc[:,7:].quantile(0.25).values
    jumlah_fitur = df_fiks.iloc[:,7:].shape[1]
    for i in range(len(df_fiks)):
      agreeIn = (df_fiks.iloc[[i]].iloc[:,7:].values[0] < treshhold).astype(int).sum()
      number_form.append(df_fiks.iloc[[i]].iloc[:,7:].values[0].mean())
      isDrop.append(agreeIn < jumlah_fitur)

    df_fiks['DropFromDf'] = np.array(isDrop)
    self.fitur9['fitur9'] = np.array(number_form)
    # df_proccess = df_fiks[df_fiks['DropFromDf']==True].reset_index(drop=True)

    return df_fiks

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def grafCreate(self,df,textData,sim=0.8):
    df_graf = {}
    df_sim = {}
    for i in range(len(df)):
      kal1 = df.iloc[[i]][textData].values[0]
      for j in range(len(df)):
        kal2 = df.iloc[[j]][textData].values[0]
        similarity = self.jaccard_similarity(kal1,kal2)

        if 'Document Number '+str(df.iloc[[j]]['doc_id'].values[0]) not in df_sim.keys():
          df_sim['Document Number '+str(df.iloc[[j]]['doc_id'].values[0])] = [(0,0) for k in range(len(df))]

        if similarity>sim:
          if df_sim['Document Number '+str(df.iloc[[j]]['doc_id'].values[0])][i][1] < similarity:
            df_sim['Document Number '+str(df.iloc[[j]]['doc_id'].values[0])][i] = (df.iloc[[j]]['teks_id'].values[0],similarity)

        if str(j) not in df_graf.keys():
          df_graf[str(j)] = []

        if similarity==1:
          df_graf[str(j)].append(0.99)
        else:
          df_graf[str(j)].append(similarity)
    return df_graf, df_sim

  def get_best_inflitation(self,df_graf):
    best_inflitation = 0
    best_Q = -999
    network = nx.from_numpy_matrix(df_graf.values)

    matrix = nx.to_scipy_sparse_matrix(network)
    for inflation in [i / 10 for i in range(11, 24)]:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)
        if Q>best_Q:
          best_inflitation = inflation
          best_Q = Q
    result = mc.run_mcl(matrix,inflation=best_inflitation)
    clusters = mc.get_clusters(result)
    return clusters

  def transform(self,df):
    df_fiks = self.extrackFitur.transform(df)
    try:
      self.fitur1 = self.extrackFitur.fitur1
    except:
      self.fitur1 = self.extrackFitur.df_clean

    try:
      self.fitur2 = self.extrackFitur.fitur2
    except:
      self.fitur2 = self.extrackFitur.df_clean
    
    try:
      self.fitur3 = self.extrackFitur.fitur3
    except:
      self.fitur3 = self.extrackFitur.df_clean

    try:
      self.fitur4 = self.extrackFitur.fitur4
    except:
      self.fitur4 = self.extrackFitur.df_clean

    try:  
      self.fitur5 = self.extrackFitur.fitur5
    except:
      self.fitur5 = self.extrackFitur.df_clean

    try:  
      self.fitur6 = self.extrackFitur.fitur6
    except:
      self.fitur6 = self.extrackFitur.df_clean

    try:  
      self.fitur7 = self.extrackFitur.fitur7
    except:
      self.fitur7 = self.extrackFitur.df_clean

    try:  
      self.fitur8 = self.extrackFitur.fitur8
    except:
      self.fitur8 = self.extrackFitur.df_clean

    try:  
      self.fitur9 = self.extrackFitur.fitur9
    except:
      self.fitur9 = self.extrackFitur.df_clean

    df_fiks = self.eliminasi_treshhold(df_fiks.replace({True: 1, False: 0}))

    g,s = self.grafCreate(df_fiks,'cleanTeks')

    df_graf = pd.DataFrame(g)
    df_fiks = pd.concat([df_fiks,pd.DataFrame.from_dict(s)],axis=1)

    #Cluster 1
    clusters = self.get_best_inflitation(df_graf)
    df_fiks['Cluster1'] = None
    for i in range(len(clusters)):
      for j in range(len(clusters[i])):
        df_fiks['Cluster1'][clusters[i][j]] = i

    NC = len(clusters)

    #Sinonim Conversion
    if self.sin_conv:
      resp = requests.get('https://raw.githubusercontent.com/victoriasovereigne/tesaurus/master/dict.json')
      resp_dict = resp.json()
      Sinonim1 = []
      for j in range(len(df_fiks)):
        kal = df_fiks.iloc[[j]].cleanTeks.values[0]
        newKal = []
        for k in kal.split(" "):
          try:
            k = resp_dict[k]['sinonim'][0]
          except:
            pass
          newKal.append(k)
        Sinonim1.append(" ".join(newKal))
      #Hasil klaster kalimat (Pertama)
      df_fiks['Sinonim1'] = np.array(Sinonim1)
    else:
      df_fiks['Sinonim1'] = df_fiks.cleanTeks

    #Cluster 2
    df_fiks2 = None
    df_to_TSO = None
    if self.markov:
      for clus in range(NC):
        try:
          df_reclus = df_fiks[df_fiks['Cluster1']==clus].copy()
          g,s = self.grafCreate(df_reclus,'Sinonim1')
          df_graf = pd.DataFrame(g)
          df_reclus['Cluster2'] = 0

          #Pengembangan Selanjutnya Untuk Pengujian Treshold
          df_reclus['TresholdMCL'] = df_graf.mean().values < df_graf.mean().quantile(0.75)
          
          clusters = self.get_best_inflitation(df_graf)
          for i in range(len(clusters)):
            for j in range(len(clusters[i])):
              df_reclus['Cluster2'][clusters[i][j]] = i
        except:
          df_reclus['Cluster2']=0
        #Hasil klaster kalimat (Kedua)
        df_fiks2 = pd.concat([df_fiks2,df_reclus])
      return df_fiks2.fillna(0)
    else:
      #Hasil akhir penanganan Redundasi
      return df_fiks.fillna(0)

class Tuna_Swamp_Optimizer:
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=[], num_markov = 8):
    self.raw_df = dataframe
    
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch     = epoch #T_Max
    self.df        = dataframe #Hasil ringkasann(TSO)
    if markov:
      self.condition  = 3
      if num_markov == 9:
        self.condition = 2 
    else:
      self.condition = 1

    self.fitur     = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.w         = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history = {}
    self.f_history = {}
    self.z         = z #Nilai Z
    self.alpha     = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    try:
      self.text_sample.remove('')
    except:
      pass

    self.w_best      = w_best
    self.f_best      = f_best
    self.result_best = []

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
      for tuna,w in enumerate(self.w):
        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten().astype("float"))
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        # fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        fitnes,tso_result = self.transform(df_new = self.raw_df.copy(),weigth = w)
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(tuna)] = [fitnes]

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)


      self.a1 = self.alpha+(1-self.alpha)*(epoch/self.epoch)
      self.a2 = (1-self.alpha)-(1-self.alpha)*(epoch/self.epoch)
      self.p  = (1-(epoch/self.epoch))**(epoch/self.epoch)

      for tuna,w in enumerate(self.w):
        new_weight = []
        for w in range(len(self.w[tuna])):
          if random.random() < self.z:
            new_weight.append(random.random()*(max(self.w[tuna])-min(self.w[tuna]))+min(self.w[tuna]))
          else:
            rand_val = random.random()
            if random.random()<0.5:
              new_weight.append(rand_val+random.random()*(rand_val-self.w_best[w])+random.choice([-1,1])*self.p**2*(rand_val-self.w_best[w]))
            else:
              #NIlai B
              valx = self.w_best[w-1]
              beta = random.random()
              if epoch == 1:
                valx = self.w_best[1]
              if (epoch/self.epoch)<random.random():
                new_weight.append(self.a1*(rand_val+beta*abs(rand_val-self.w_best[w]))+self.a2*valx)
              else:
                new_weight.append(self.a1*(max(self.w_best)+beta*abs(max(self.w_best)-self.w_best[w]))+self.a2*valx)
        self.w[tuna] = new_weight

    long_text_test = len(self.text_sample)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

    # _,tso_result = self.transform(self.raw_df.copy())
    # text,result_table = self.sortResult(tso_result,long_text_test)
    # self.text_best = text
    # self.f_best = self.similarity_check_rogue(". ".join(self.text_sample),text,'rouge2')

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]

    print(fitur.columns)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    
    # fitnes = []
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))

    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class TSO_Multi_Doc(PreprocessTuna,Tuna_Swamp_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe
    self.TsoModel  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Preprocessing_Transform":[],
      "Model_Transform":[]
    }

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="",preprocessing = True):
    for t in self.topic:
      self.TsoModel['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      
      global PreProcessDataTrain
      if preprocessing: 
        preptuna = PreprocessTuna(df=data, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain = preptuna
      else:
        try:
          preptuna = PreProcessDataTrain
        except:
          preptuna = PreprocessTuna(df=data, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
          PreProcessDataTrain = preptuna
      
      self.TsoModel['Preprocessing_Transform'].append(preptuna)

      data     = data['ringkasan'].values[0]
      self.TsoModel['Ringkasan_Sample'].append(data)
      tunaswam = Tuna_Swamp_Optimizer(epoch,tuna,preptuna.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z)
      tunaswam.fit()

      self.TsoModel['Model_Transform'].append(tunaswam)

  def transform(self,df):
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    topik_data = pd.DataFrame.from_dict(self.TsoModel)

    for index in range(len(topik_data)):
      data = topik_data.iloc[[index]]

      preptuna = data['Preprocessing_Transform'].values[0]
      tunaswam = data['Model_Transform'].values[0]

      df_new = preptuna.transform(df)
      fitnes,tso_result = tunaswam.transform(df_new)
      text,result_table = tunaswam.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict

class TSO_Multi_DocCombine(PreprocessTuna,Tuna_Swamp_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.TsoModel  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing = True,w_best = [], f_best = 0.0,num_markov=8):

    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.TsoModel['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.TsoModel['Ringkasan_Sample'].append(data)
      tunaswam = Tuna_Swamp_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      tunaswam.fit()

      w_best = tunaswam.w_best
      f_best = tunaswam.f_best

      self.TsoModel['Model_Transform'].append(tunaswam)

      self.Model_Group = tunaswam

  def transform(self,df,k,weigth):
    self.df_fit   = df
    tunaswam = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.TsoModel)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = tunaswam.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = tunaswam.sortResult(tso_result)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict


class Bat_Optimizer:
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=[], num_markov = 8):
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch      = epoch #T_Max
    self.df         = dataframe #Hasil ringkasann(TSO)
    self.condition  = 3
    if num_markov == 9:
      self.condition = 0 
    self.fitur      = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.NP, self.D = self.fitur.shape
    self.w          = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history  = {}
    self.f_history  = {}
    self.z          = z #Nilai Z
    self.alpha      = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    self.tuna        = tuna

    try:
      self.text_sample.remove('')
    except:
      pass

    self.w_best      = w_best
    self.f_best      = f_best
    self.result_best = []

  def Fun(self, D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
        if len(self.w_best) == 0:
          minimum = 0.5
          maximum = 0.5
        else:
          minimum = min(self.w_best)
          maximum = max(self.w_best)

        Algorithm     = BatAlgorithm(self.D, self.NP, self.tuna, 0.5, 0.5, 0.0, 1.0, -10.0, 10.0, self.Fun)
        Algorithm.v   = self.fitur.values.tolist()
        Algorithm.Sol = self.fitur.values.tolist()

        Algorithm.best_bat()

        w             = np.array(Algorithm.best)
        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten())
          print(w)
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(self.tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = [fitnes]

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]
    print(fitur.shape)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    
    # fitnes = []
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')
    
    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class Bat_Multi_DocCombine(PreprocessTuna,Bat_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.Model  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing=True, w_best = [], f_best = 0.0,num_markov=8):
    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        print("sukses mengambil tempat")
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        print("gagal mengambil tempat")
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.Model['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.Model['Ringkasan_Sample'].append(data)
      bat      = Bat_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      bat.fit()

      w_best = bat.w_best
      f_best = bat.f_best

      self.Model['Model_Transform'].append(bat)

      self.Model_Group = bat #Menyimpan Model Grup Terbaik

  def transform(self,df,k,weigth):
    self.df_fit   = df
    bat      = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = bat.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict
  
class IWO_Optimizer:
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=0):
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch      = epoch #T_Max
    self.df         = dataframe #Hasil ringkasann(TSO)
    self.condition  = 3
    if num_markov == 9:
      self.condition = 0 
    self.fitur      = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.w          = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history  = {}
    self.f_history  = {}
    self.z          = z #Nilai Z
    self.alpha      = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    self.tuna       = tuna

    try:
      self.text_sample.remove('')
    except:
      pass

    if len(w_best) == 0:
      self.w_best    = [random.random() for i in range(len(self.fitur.columns))]
    else:
      self.w_best    = w_best

    self.f_best      = f_best
    self.result_best = []

  def obj(self, x):
      """
      The sphere function
      :param x:
      :return:
      """

      num = 0
      for i in range(len(x)):
          num += x[i] ** 2
      return num


  def boundary_check(self, x, lb, ub, dim):
      """
      Check the boundary
      :param x: a candidate solution
      :param lb: the lower bound (list)
      :param ub: the upper bound (list)
      :param dim: dimension
      :return:
      """
      for i in range(dim):
          if x[i] < lb[i]:
              x[i] = lb[i]
          elif x[i] > ub[i]:
              x[i] = ub[i]
      return x


  def IWOAlgorithm(self, ipop = 20, mpop = 100, iter = 2000, smin = 0, smax = 5, isigma = 1, fsigma = 1e-6, pos = []):
      """
      The main function of the IWO
      :param ipop: The initial population size
      :param mpop: The maximum population size
      :param iter: The maximum number of iterations
      :param smin: The minimum number of seeds
      :param smax: The maximum number of seeds
      :param isigma: The initial value of standard deviation
      :param fsigma: The final value of standard deviation
      :param lb: The lower bound (list)
      :param ub: The upper bound (list)
      :return:
      """
      print(pos)
      ipop = len(pos)
      lb = [-10] * len(pos)
      ub = [10] * len(pos)
      # Step 1. Initialization
      dim = len(lb)  # dimension
      pos = pos  # the position of weeds
      score = []  # the score of weeds
      for _ in range(ipop):
          score.append(self.obj(pos[-1]))
      gbest = min(score)  # the global best
      gbest_pos = pos[score.index(gbest)].copy()  # the global best individual
      iter_best = []  # the global best of each iteration
      con_iter = 0  # the convergence iteration

      # Step 2. The main loop
      for t in range(iter):

          # Step 2.1. Update standard deviation
          sigma = ((iter - t - 1) / (iter - 1)) ** 2 * (isigma - fsigma) + fsigma

          # Step 2.2. Reproduction
          new_pos = []
          new_score = []
          min_score = min(score)
          max_score = max(score)
          for i in range(len(pos)):
              try:
                ratio = (score[i] - max_score) / (min_score - max_score)
              except:
                ratio = (score[i] - max_score) / 2
              snum = math.floor(smin + (smax - smin) * ratio)  # the number of seeds

              for _ in range(snum):
                  temp_pos = [pos[i][j] + random.gauss(0, sigma) for j in range(dim)]
                  temp_pos = self.boundary_check(temp_pos, lb, ub, dim)
                  new_pos.append(temp_pos)
                  new_score.append(self.obj(temp_pos))

          # Step 2.3. Competitive exclusion
          new_pos.extend(pos)
          new_score.extend(score)

          if len(new_pos) > mpop:
              pos = []
              score = []
              sorted_index = np.argsort(new_score)
              for i in range(mpop):
                  pos.append(new_pos[sorted_index[i]])
                  score.append(new_score[sorted_index[i]])
          else:
              pos = new_pos
              score = new_score

          # Step 2.4. Update the global best
          if min(score) < gbest:
              gbest = min(score)
              gbest_pos = pos[score.index(gbest)]
              con_iter = t + 1
          iter_best.append(gbest)

      # Step 3. Sort the results
      x = [i for i in range(iter)]
      iter_best = [math.log10(iter_best[i]) for i in range(iter)]
      return {'best score': gbest, 'best solution': gbest_pos, 'convergence iteration': con_iter}

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
        if len(self.w_best) == 0:
          minimum = 0.5
          maximum = 0.5
        else:
          minimum = min(self.fitur.values.tolist())
          maximum = max(self.fitur.values.tolist())

        Algorithm     = self.IWOAlgorithm(pos = self.fitur.values.tolist())

        w             = np.array(Algorithm['best solution'])
        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten())
          print(w)
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(self.tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = [fitnes]
        print(fitnes)

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]
    print(fitur.shape)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    # fitnes = []
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')

    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class IWO_Multi_DocCombine(PreprocessTuna,IWO_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.Model  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing = True, w_best = [], f_best = 0.0,num_markov=8):

    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.Model['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.Model['Ringkasan_Sample'].append(data)
      bat      = IWO_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      bat.fit()

      w_best = bat.w_best
      f_best = bat.f_best

      self.Model['Model_Transform'].append(bat)

      self.Model_Group = bat #Menyimpan Model Grup Terbaik

  def transform(self,df,k,weigth):
    self.df_fit   = df
    bat      = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = bat.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict
  
class PSO_Optimizer:
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=[], num_markov = 8):
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch      = epoch #T_Max
    self.df         = dataframe #Hasil ringkasann(TSO)
    self.condition  = 3
    if num_markov == 9:
      self.condition = 0 
    self.fitur      = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.w          = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history  = {}
    self.f_history  = {}
    self.z          = z #Nilai Z
    self.alpha      = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    self.tuna       = tuna

    try:
      self.text_sample.remove('')
    except:
      pass

    if len(w_best) == 0:
      self.w_best    = [random.random() for i in range(len(self.fitur.columns))]
    else:
      self.w_best    = w_best

    self.f_best      = f_best
    self.result_best = []

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
        if len(self.w_best) == 0:
          minimum = 0.5
          maximum = 0.5
        else:
          minimum = min(self.w_best)
          maximum = max(self.w_best)

        initial   = self.w_best
        bounds    = [(-10,10)]*len(initial)

        Algorithm = pso_simple.minimize(sphere, initial, bounds, num_particles=15, maxiter=30, verbose=True)
        w         = np.array(Algorithm[1])

        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten())
          print(w)
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(self.tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = [fitnes]

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]
    print(fitur.shape)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    # fitnes = []
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')

    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class PSO_Multi_DocCombine(PreprocessTuna,PSO_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.Model  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing=True, w_best = [], f_best = 0.0,num_markov=8):
    
    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.Model['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.Model['Ringkasan_Sample'].append(data)
      bat      = PSO_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      bat.fit()

      w_best = bat.w_best
      f_best = bat.f_best

      self.Model['Model_Transform'].append(bat)

      self.Model_Group = bat #Menyimpan Model Grup Terbaik

  def transform(self,df,k,weigth):
    self.df_fit   = df
    bat      = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = bat.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict
  
class ABC_Optimizer:
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=[], num_markov = 8):
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch      = epoch #T_Max
    self.df         = dataframe #Hasil ringkasann(TSO)
    self.condition  = 3
    if num_markov == 9:
      self.condition = 0 
    self.fitur      = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.w          = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history  = {}
    self.f_history  = {}
    self.z          = z #Nilai Z
    self.alpha      = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    self.tuna       = tuna

    try:
      self.text_sample.remove('')
    except:
      pass

    if len(w_best) == 0:
      self.w_best    = [random.random() for i in range(len(self.fitur.columns))]
    else:
      self.w_best    = w_best

    self.f_best      = f_best
    self.result_best = []

  def minimize_integers(self,integers):
    return sum(integers)

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
        if len(self.w_best) == 0:
          minimum = 0.5
          maximum = 0.5
        else:
          minimum = min(self.w_best)
          maximum = max(self.w_best)

        w = []
        for index,ftr in enumerate(self.fitur.keys()):
          abc = ABC(10, self.minimize_integers)

          for i_ftr,ftr_val in enumerate(self.fitur[ftr].values.tolist()):
            if ftr_val>0:
              print(ftr_val*self.w_best[index]*10)
              abc.add_param(0, round(ftr_val*self.w_best[index]*100), name=f'Int_{i_ftr}')
            else:
              print(self.w_best[index]*10)
              abc.add_param(0, round(self.w_best[index]*100), name=f'Int_{i_ftr}')

          abc.initialize()
          print("initialize")
          abc.search()
          print("search")
          w.append(abc.best_fitness)

        w         = np.array(w)

        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten())
          print(w)
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(self.tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = [fitnes]

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]
    print(fitur.shape)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    # fitnes = []
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')

    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class ABC_Multi_DocCombine(PreprocessTuna,ABC_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.Model  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing=True, w_best = [], f_best = 0.0,num_markov=8):
    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.Model['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.Model['Ringkasan_Sample'].append(data)
      bat      = ABC_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      bat.fit()

      w_best = bat.w_best
      f_best = bat.f_best

      self.Model['Model_Transform'].append(bat)

      self.Model_Group = bat #Menyimpan Model Grup Terbaik

  def transform(self,df,k,weigth):
    self.df_fit   = df
    bat      = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = bat.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict
  
class WOA:
  def __init__(self, obj_func, n_whale, spiral_constant, n_iter,
              lb, ub):
      self.obj_func = obj_func
      self.n_whale = n_whale
      self.spiral_constant = spiral_constant
      self.n_iter = n_iter
      self.lb = lb
      self.ub = ub
      self.whale = {}
      self.prey = {}

  def init_whale(self):
      tmp = [np.random.uniform(self.lb, self.ub, size=(len(self.lb),))
              for i in range(self.n_whale)]
      self.whale['position'] = np.array(tmp)
      self.whale['fitness'] = self.obj_func(self.whale['position'])

  def init_prey(self):
      tmp = [np.random.uniform(self.lb, self.ub, size=(len(self.lb),))]
      self.prey['position'] = np.array(tmp)
      self.prey['fitness'] = self.obj_func(self.prey['position'])

  def update_prey(self):
      if self.whale['fitness'].min() < self.prey['fitness'][0]:
          self.prey['position'][0] = self.whale['position'][self.whale['fitness'].argmin()]
          self.prey['fitness'][0] = self.whale['fitness'].min()

  def search(self, idx, A, C):
      random_whale = self.whale['position'][np.random.randint(low=0, high=self.n_whale,
                                                              size=len(idx[0]))]
      d = np.abs(C[..., np.newaxis] * random_whale - self.whale['position'][idx])
      self.whale['position'][idx] = np.clip(random_whale - A[..., np.newaxis] * d, self.lb, self.ub)

  def encircle(self, idx, A, C):
      d = np.abs(C[..., np.newaxis] * self.prey['position'] - self.whale['position'][idx])
      self.whale['position'][idx] = np.clip(self.prey['position'][0] - A[..., np.newaxis] * d, self.lb, self.ub)

  def bubble_net(self, idx):
      d_prime = np.abs(self.prey['position'] - self.whale['position'][idx])
      l = np.random.uniform(-1, 1, size=len(idx[0]))
      self.whale["position"][idx] = np.clip(
          d_prime * np.exp(self.spiral_constant * l)[..., np.newaxis] * np.cos(2 * np.pi * l)[..., np.newaxis]
          + self.prey["position"],
          self.lb,
          self.ub,
      )

  def optimize(self, a):

      p = np.random.random(self.n_whale)
      r1 = np.random.random(self.n_whale)
      r2 = np.random.random(self.n_whale)
      A = 2 * a * r1 - a
      C = 2 * r2
      search_idx = np.where((p < 0.5) & (abs(A) > 1))
      encircle_idx = np.where((p < 0.5) & (abs(A) <= 1))
      bubbleNet_idx = np.where(p >= 0.5)
      self.search(search_idx, A[search_idx], C[search_idx])
      self.encircle(encircle_idx, A[encircle_idx], C[encircle_idx])
      self.bubble_net(bubbleNet_idx)
      self.whale['fitness'] = self.obj_func(self.whale['position'])

  def run(self):
      self.init_whale()
      self.init_prey()
      f_values = [self.prey['fitness'][0]]
      for n in range(self.n_iter):
          #print("Iteration = ", n, " f(x) = ", self.prey['fitness'][0])
          a = 2 - n * (2 / self.n_iter)
          self.optimize(a)
          self.update_prey()
          f_values.append(self.prey['fitness'][0])
      optimal_x = self.prey['position'].squeeze()
      return f_values, optimal_x

class Whale_Optimizer(WOA):
  def __init__(self,epoch,tuna,dataframe,text_sample,sin_conv=True,treshhold = True, markov = True,a = random.random(),z = random.random(),w_best = [],f_best=[], num_markov = 8):
    if treshhold:
      dataframe = dataframe[dataframe['DropFromDf']==True].reset_index(drop=True)
    if markov:
      dataframe = dataframe[dataframe['TresholdMCL']==True].reset_index(drop=True)

    self.epoch      = epoch #T_Max
    self.df         = dataframe #Hasil ringkasann(TSO)
    self.condition  = 3
    if num_markov == 9:
      self.condition = 0 
    self.fitur      = dataframe.iloc[:,7:-(len(dataframe['doc_id'].unique())+2)-self.condition].copy()
    self.w          = [np.array([random.random() for i in range(len(self.fitur.columns))]) for x in range(tuna)]
    self.w_history  = {}
    self.f_history  = {}
    self.z          = z #Nilai Z
    self.alpha      = a #Nilai A
    self.a1,self.a2,self.p = None,None,None
    self.text_sample = [t.strip() for t in text_sample.split(". ")]
    self.tuna       = tuna

    try:
      self.text_sample.remove('')
    except:
      pass

    if len(w_best) == 0:
      self.w_best    = [random.random() for i in range(len(self.fitur.columns))]
    else:
      self.w_best    = w_best

    self.f_best      = f_best
    self.result_best = []

  def minimize_integers(self,integers):
    return sum(integers)

  def jaccard_similarity(self,x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

  def similarity_check_rogue(self,x,y,var_ext):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return scores[var_ext].precision

  def f(self,X):
    A = 10
    sol = []
    for ind in X:
        sol.append(A*len(ind) + sum([(i**2 - A * np.cos(2 * np.pi * i)) for i in ind]) )
    return np.array(sol)

  def fit(self):
    for epoch in range(self.epoch):
      #refisi Tuna
        w = []

        lb = np.array(self.w_best)-10
        ub = np.array(self.w_best)+10

        woa = WOA(self.f, 100, 0.5, 100, lb, ub)
        f_values, optimal_x = woa.run()

        w         = np.array(optimal_x)

        epoch_record = []
        for data in range(len(self.df)):
          print("===========================")
          print(self.fitur.loc[[data]].values.flatten())
          print(w)
          epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*w))

        self.df['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = np.array(epoch_record)
        print('epoch_'+str(epoch)+'_tuna_'+str(self.tuna))
        self.w_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = w

        long_text_test = len(self.text_sample)

        text_result = ''
        #if self.order:
        text_result = self.df.sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()
        # else:
        #   text_result = self.df.sample(n=long_text_test).sort_values(by=['epoch_'+str(epoch)+'_tuna_'+str(tuna)],ascending=False)[:long_text_test]['teks'].values.tolist()

        fitnes = self.similarity_check_rogue(". ".join(self.text_sample),". ".join(text_result),'rouge2')
        self.f_history['epoch_'+str(epoch)+'_tuna_'+str(self.tuna)] = [fitnes]

        if len(self.w_best) == 0:
          self.w_best = w
          self.f_best = fitnes
          self.result_best = '. '.join(text_result)
        else:
          #Revisi Weigth dan Fitness
          if fitnes>self.f_best:
            self.w_best = w
            self.f_best = fitnes
            self.result_best = '. '.join(text_result)

    epoch_record = []
    for data in range(len(self.df)):
      epoch_record.append(sum(self.fitur.loc[[data]].values.flatten()*self.w_best))
    self.df['final_point'] = np.array(epoch_record)

  def transform(self,df_new,k=0,weigth=[]):
    if len(weigth) == 0:
      weigth = self.w_best
    epoch_record = []
    fitur = df_new.iloc[:,7:-(len(df_new['doc_id'].unique())+2)-self.condition]
    print(fitur.shape)
    for data in range(len(df_new)):
      epoch_record.append(sum(fitur.loc[[data]].values.flatten()*weigth))
    df_new['final_point'] = np.array(epoch_record)

    long_text_test = k
    if k==0:
      long_text_test = len(self.text_sample)

    text_result_clean = df_new.sort_values(by=['final_point'],ascending=False)[:long_text_test]['cleanTeks'].values.tolist()
    text_result       = self.sortResult(df_new,long_text_test)[0].split(". ")
    # fitnes = []
    # for text in range(long_text_test):
    #   fitnes.append(self.similarity_check_rogue(self.text_sample[text],text_result[text],'rouge2'))
    fitnes = self.similarity_check_rogue(". ".join(self.text_sample[:long_text_test]),". ".join(text_result[:long_text_test]),'rouge2')

    return fitnes,df_new

  def sortResult(self,urut,lenText):
    useData = urut.loc[:,['title','teks','final_point']+[i for i in urut.keys() if 'Document Number ' in i]]
    result_df = useData.sort_values(by=['final_point'],ascending=False).iloc[:lenText,:]
    sum_point = []
    for r in range(len(result_df)):
      data = result_df.iloc[[r]].loc[:,[i for i in urut.keys() if 'Document Number ' in i]]

      listPoint = []
      for val in data.values[0]:
        if 'tuple' not in str(type(val)):
          val = re.sub('[\(\)]','',val).split(',')
          val = (int(val[0]),float(val[1]))
        if val[1]<0:
          continue
        listPoint.append((1/(val[0]+1))*val[1])

      sum_point.append(np.mean(listPoint))
    result_df['sum_point'] = np.array(sum_point)
    return ". ".join(result_df.sort_values(by=['sum_point'],ascending=False)['teks'].values.tolist()),result_df

class Whale_Multi_DocCombine(PreprocessTuna,Whale_Optimizer):
  def __init__(self,dataframe):

    dataframe      = dataframe.dropna()
    dataframe      = dataframe.reset_index(drop=True)

    self.topic     = list(dict.fromkeys(dataframe.Topik.values.tolist()))
    self.dataframe = dataframe

    self.Preprocessing_Transform = None

    self.Model  = {
      "Topik_Name":[],
      "Ringkasan_Sample":[],
      "Model_Transform":[]
    }

    self.Model_Group = None

  def fit(self,sin_conv,treshhold,markov,epoch,tuna,a,z,tfidf=True,tfidf_compress="", preprocessing=True, w_best = [], f_best = 0.0,num_markov=8):
    global PreProcessDataTrain
    if preprocessing: 
      self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
      PreProcessDataTrain =  self.Preprocessing_Transform
    else:
      try:
        self.Preprocessing_Transform = PreProcessDataTrain
      except:
        self.Preprocessing_Transform = PreprocessTuna(df = self.dataframe, sin_conv=sin_conv,treshhold = treshhold,markov = markov,tfidf = tfidf,tfidf_compress = tfidf_compress)
        PreProcessDataTrain =  self.Preprocessing_Transform

    for t in self.topic: #Membaca Data tiap Topik
      self.Model['Topik_Name'].append(t)

      data     = self.dataframe[self.dataframe['Topik']==t].copy().reset_index(drop=True)
      data     = data['ringkasan'].values[0]
      self.Model['Ringkasan_Sample'].append(data)
      bat      = Whale_Optimizer(epoch,tuna,self.Preprocessing_Transform.df_to_TSO,data,sin_conv=sin_conv,treshhold = treshhold, markov = markov,a = a,z = z,w_best=w_best,f_best=f_best,num_markov=num_markov)
      bat.fit()

      w_best = bat.w_best
      f_best = bat.f_best

      self.Model['Model_Transform'].append(bat)

      self.Model_Group = bat #Menyimpan Model Grup Terbaik

  def transform(self,df,k,weigth):
    self.df_fit   = df
    bat      = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = k, weigth = weigth)
      text,result_table = bat.sortResult(tso_result,10)

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)

      scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
      scores = scorer.score(data['Ringkasan_Sample'].values[0],text)
      result_dict["rouge1"].append(scores['rouge1'].precision)
      result_dict["rouge2"].append(scores['rouge2'].precision)
      result_dict["rouge3"].append(scores['rouge3'].precision)

    return result_dict

class Transform_All_Algorithm:
  def __init__(self,transformClass):
    self.Preprocessing_Transform = transformClass.Preprocessing_Transform
    self.Model_Group             = transformClass.Model_Group
    self.df_fit                  = None
    try:
      self.Model                   = transformClass.TsoModel
    except:
      self.Model                   = transformClass.Model
  
  def changeModel(self,transformClass):
    self.Model_Group             = transformClass.Model_Group
    try:
      self.Model                   = transformClass.TsoModel
    except:
      self.Model                   = transformClass.Model

  def get_coherence(self,pred):
    print(len(pred))
    cosins = []
    for p in range(1,len(pred)):
      vectors = TfidfVectorizer().fit_transform([pred[p-1], pred[p]])
      similarity = cosine_similarity(vectors[0], vectors[1])
      print(similarity)
      cosins.append(similarity[0][0])
    print(len(cosins))
    return sum(cosins)/len(pred)

  def eliminasi_treshhold(self,df_fiks):
    number_form = []
    treshhold    = df_fiks.iloc[:,7:].quantile(0.25).values
    jumlah_fitur = df_fiks.iloc[:,7:].shape[1]
    for i in range(len(df_fiks)):
      agreeIn = (df_fiks.iloc[[i]].iloc[:,7:].values[0] < treshhold).astype(int).sum()
      number_form.append(df_fiks.iloc[[i]].iloc[:,7:].values[0].mean())

    return np.mean(number_form)
  
  def similarity_check_rogue(self,x,y):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)
    scores = scorer.score(x,y)

    return {
      'rouge1':scores['rouge1'].precision,
      'rouge2':scores['rouge2'].precision,
      'rouge3':scores['rouge3'].precision,
      'rougeL':scores['rougeL'].precision
    }

  def transform(self,df,length_result=0,preprocessing = True, weigth=[], answer_form = "standard"):
    global DataFrame_Fit
    if preprocessing:
      print("Preprocessing Running")
      self.df_fit   = self.Preprocessing_Transform.transform(df)
      DataFrame_Fit = self.df_fit
    else:
      try:
        print("Using Save Data")
        self.df_fit   = DataFrame_Fit
      except:
        self.df_fit   = self.Preprocessing_Transform.transform(df)
        DataFrame_Fit = self.df_fit
    bat           = self.Model_Group #Mengambil Model Terbaik Yang sebelumnya disimpan

    #Deklarasi Variabel Penyimpan Hasil per Topik
    result_dict = {
        "Topik_Name"   :[],
        "rouge1"    :[],
        "rouge2"    :[],
        "rouge3"    :[],
        "rougeL"    :[],
        "coherence_by_sort" :[],
        "coherence_by_point" :[],
        "coherence_by_datetime" :[],
        "Result"       :[],
        "Table_Result" :[],
        "fitnes"       :[],
        "Opt_Result"   :[],
        "Weight":[],
        "Refrence"     :[],
        "Redudansi"     :[]
    }

    #Pengambilan Data Ringkasan Pakar Per Topik
    topik_data = pd.DataFrame.from_dict(self.Model)

    for index in range(len(topik_data)): #Melakukan Peringkasan dan Pengujian hasil per ringkasan pakar
      data = topik_data.iloc[[index]]

      print("Start Transforming")
      if len(weigth) != 0:
        weigth = np.array(weigth)
        print("Sudah Masuk")
      fitnes,tso_result = bat.transform(self.df_fit.copy(),k = length_result, weigth = weigth)
      print("End Transforming")

      if length_result==0:
        length_result = len(data['Ringkasan_Sample'].values[0].split(". "))
      
      text = None
      result_table = None
      
      text_sort,result_table = bat.sortResult(tso_result,length_result)
      
      result_df = tso_result.sort_values(by=['final_point'],ascending=False)
      text_point = ". ".join(result_df[:length_result]['teks'].values.tolist())

      result_dict["Topik_Name"].append(data['Topik_Name'].values[0])
      result_dict["Result"].append(text_sort)
      result_dict["fitnes"].append(fitnes)
      result_dict["Table_Result"].append(result_table)
      result_dict["Opt_Result"].append(tso_result)

      # for t in range(length_result):
      scores = self.similarity_check_rogue(data['Ringkasan_Sample'].values[0],text_point)
      result_dict["Refrence"].append(data['Ringkasan_Sample'].values[0])
      result_dict["Weight"].append(weigth)

      result_dict["rouge1"].append(scores['rouge1'])
      result_dict["rouge2"].append(scores['rouge2'])
      result_dict["rouge3"].append(scores['rouge3'])
      result_dict["rougeL"].append(scores['rougeL'])
      result_dict["coherence_by_sort"].append(self.get_coherence(text_sort.split(". ")))
      result_dict["coherence_by_point"].append(self.get_coherence(text_sort.split(". ")))
      result_dict["Redudansi"].append(self.eliminasi_treshhold(tso_result.iloc[:,:6+8]))

    return result_dict