from __future__ import division 
import codecs
import random
import re
import math
import os
import json

stopwords=set(codecs.open('stopwords.txt','r','utf-8').readlines())

def createVocab(path):
	#path='./data'
	vocabulary=set([])
	# stopwords=set(codecs.open('stopwords.txt','r','utf-8').readlines())
	for filename in os.listdir(path):
		vocabulary=vocabulary | set(codecs.open(path+'/'+filename,'r','utf-8').read().split())
	return list(vocabulary-stopwords)

def prepareData(path,vocabulary,ratio=0.7):
	trainData=[]
	testData=[]
	lables=[]
	numRe=re.compile('\d+')
	for filename in os.listdir(path):
		x=codecs.open(path+'/'+filename,'r','utf-8').read().split()
		x=[word for word in x if word not in stopwords]
		# y,num=filename.split('.')[0].replace(numRe,' ')
		y,num=numRe.subn('',filename.split('.')[0])
		data=[x,y]
		lables.append(y)
		if random.random()<ratio:
			trainData.append(data)
		else:
			testData.append(data)
	return trainData,testData

class NaiveBayes():
	"""Naive Bayes for news clssify"""
	def __init__(self):
		self.freq={}
		self.disappear={}
		self.lables=[]
		self.prior={}
		

		
	def train(self,vocabulary,trainSet,smooth=1):
		self.lables=list(set([data[1] for data in trainSet]))
		# self.lables=['auto', 'business', 'sports']
		n=len(vocabulary)#
		i=0;
		for lable in self.lables:
			self.freq[lable]={}
			self.disappear[lable]={}
			self.prior[lable]=0
		for data in trainSet:
			i+=1
			if(i%50==0):
				print (i)
			self.prior[data[1]]+=1
			for word in data[0]:
				if(word in self.freq[data[1]].keys()):
					self.freq[data[1]][word]+=1
				else:
					self.freq[data[1]][word]=smooth# laplace

		for lable in self.lables:
			total=n*smooth
			self.prior[lable]=(float)(self.prior[lable]/len(trainSet))
			for word in self.freq[lable]:
				total+=self.freq[lable][word]
			for word in self.freq[lable]:
				self.freq[lable][word]=(float)(self.freq[lable][word]/total)
			self.disappear[lable]=(float)(smooth/total)
		#self.toJson('nb.json')


	def predict(self,x):
		posteriorList=[]
		for lable in self.lables:
			# posterior={"lable":lable,"prob":math.log(self.prior[lable])}
			prob=math.log(self.prior[lable])
			uniquwords=list(set(x))
			count=0
			for word in x:
				if word in self.freq[lable].keys():
					prob+=math.log(self.freq[lable][word])
				else:
					prob+=math.log(self.disappear[lable])
			posteriorList.append((lable,prob))
		posteriorList.sort(key=lambda x:x[1])
		posteriorList.reverse()
		return posteriorList[0]

	def vali(self,testData):
		co={}
		for lable in self.lables:
			co[lable]={}
			for cl in self.lables:
				co[lable][cl]=0

		i=0
		for data in testData:
			i+=1
			if(i%20==0):
				print (i,' of ',len(testData))
			co[data[1]][self.predict(data[0])[0]]+=1
		print co

		recall={}
		for result in co:
			sum=0
			for lab in co[result]:
				sum+=co[result][lab]
			recall[result]=co[result][result]/sum
			print ('recall:',result,recall[result])
		ac={}
		col={}
		for result in co:
			col[result]=0;
		for result in co:
			for lab in co[result]:
				col[lab]+=co[result][lab]
		for result in co:
			ac[result]=co[result][result]/col[result]
			print ('ac:',result,ac[result])

		for result in co:
			print ('f1:',result,2*ac[result]*recall[result]/(ac[result]+recall[result]))

	def toJson(self,file):
		obj={"freq":self.freq,"disappear":self.disappear,"lables":self.lables,"prior":self.prior}
		f=codecs.open(file,'w','utf-8')
		f.write(json.dumps(obj,ensure_ascii=False))
		f.close()

	def fromJson(self,file):
		f=codecs.open(file,'r','utf-8')
		txt=f.read()
		f.close()
		obj=json.loads(txt)
		self.freq=obj['freq']
		self.disappear=obj['disappear']
		self.lables=obj['lables']
		self.prior=obj['prior']
		

if __name__ == '__main__':
	voc=createVocab('data');
	print('create vocabulary OK')
	trainset,testset=prepareData('data',None,0.95)
	print ('trainset,testset are created')
	nb=NaiveBayes()
	nb.fromJson('nb0001.json')
	nb.vali(testset)