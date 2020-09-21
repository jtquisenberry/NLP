# https://glowingpython.blogspot.com/2014/02/terms-selection-with-chi-square.html

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism','talk.religion.misc',
              'comp.graphics','sci.space']

posts = fetch_20newsgroups(subset='train', categories=categories,
                           shuffle=True, random_state=42,
                           remove=('headers','footers','quotes'))

# posts.data is a list of document bodies



from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=True,stop_words='english')
X = vectorizer.fit_transform(posts.data)
print(X[0].nonzero())

from sklearn.feature_selection import chi2
# compute chi2 for each feature
chi2score = chi2(X,posts.target)[0]




from pylab import barh,plot,yticks,show,grid,xlabel,figure
figure(figsize=(6,6))
wscores = zip(vectorizer.get_feature_names(),chi2score)
wchi2 = sorted(wscores,key=lambda x:x[1])

# The asterisk (*) unpacks arguments from a tuple and sends them as
# separate positional arguments.
topchi2 = zip(*wchi2[-25:])

# Deprecated 2.7 Method
# x = range(len(topchi2[1]))

# Modern 3.6 Method
# https://stackoverflow.com/questions/27431390/typeerror-zip-object-is-not-subscriptable
topchi2 = list(topchi2)
# Index 0 is a tuple of labels
# Index 1 is a tuple of scores
x = range(len(topchi2[1]))


labels = topchi2[0]
barh(x,topchi2[1],align='center',alpha=.2,color='g')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
yticks(x,labels)
xlabel('$\chi^2$')
show()

a = 1