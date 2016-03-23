# Evaluating Performance
def getConfusionMatrix(param):
	import numpy as np, pylab as pl
	from sklearn import metrics
	categories = param[0]
	y = param[1][0]
	pred = param[1][1]
	# get overall accuracy and F1 score to print at top of plot
	pscore = metrics.accuracy_score(y, pred)
	score = metrics.f1_score(y, pred, pos_label=list(set(y)))
	# get size of the full label set
	dur = len(categories)
	#print "Building testing confusion matrix..."
	# initialize score matrices
	trueScores = np.zeros(shape=(dur,dur))
	predScores = np.zeros(shape=(dur,dur))
	# populate totals
	for i in xrange(len(y)-1):
	  trueIdx = y[i]
	  predIdx = pred[i]
	  trueScores[trueIdx,trueIdx] += 1
	  predScores[trueIdx,predIdx] += 1
	# create %-based results
	trueSums = np.sum(trueScores,axis=0)
	conf = np.zeros(shape=predScores.shape)
	for i in xrange(len(predScores)):
	  for j in xrange(dur):
	    conf[i,j] = predScores[i,j] / trueSums[i]
	# plot the confusion matrix
	hq = pl.figure(figsize=(15,15));
	aq = hq.add_subplot(1,1,1)
	aq.set_aspect(1)
	res = aq.imshow(conf,cmap=pl.get_cmap('Greens'),interpolation='nearest',vmin=-0.05,vmax=1.)
	width = len(conf)
	height = len(conf[0])
	done = []
	# label each grid cell with the misclassification rates
	for w in xrange(width):
	  for h in xrange(height):
	      pval = conf[w][h]
	      c = 'k'
	      rais = w
	      if pval > 0.5: c = 'w'
	      if pval > 0.001:
	        if w == h:
	          aq.annotate("{0:1.1f}%\n{1:1.0f}/{2:1.0f}".format(pval*100.,predScores[w][h],trueSums[w]), xy=(h, w), 
	                  horizontalalignment='center',
	                  verticalalignment='center',color=c,size=10)
	        else:
	          aq.annotate("{0:1.1f}%\n{1:1.0f}".format(pval*100.,predScores[w][h]), xy=(h, w), 
	                  horizontalalignment='center',
	                  verticalalignment='center',color=c,size=10)
	# label the axes
	pl.xticks(range(width), categories[:width],rotation=90,size=10)
	pl.yticks(range(height), categories[:height],size=10)
	# add a title with the F1 score and accuracy
	aq.set_title("lbl" + " Prediction, Test Set (f1: "+"{0:1.3f}".format(score)+', accuracy: '+'{0:2.1f}%'.format(100*pscore)+", " + str(len(y)) + " items)",fontname='Arial',size=10,color='k')
	aq.set_ylabel("Actual",fontname='Arial',size=10,color='k')
	aq.set_xlabel("Predicted",fontname='Arial',size=10,color='k')
	pl.grid(b=True,axis='both')
	# save it
	pl.savefig("pred.conf.test.png")
