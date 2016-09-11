# Representing Object Association with Vector Similarity

This is an experimental investigation into a machine learning technique
for modelling associations between objects by mapping each object to
a vector in some space. These vectors are, to begin with, seeded with
random values. But by refining the vector values, the similarity of
vectors can be made to indicate the strength of association. The hope
is that after mapping objects to vectors,new associations can be inferred
from vector similarity which weren't present in the original data.

This has potential uses in [collaboative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering),
[anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) and
similar applications. Typical solutions, particulary in [information retrieval](https://en.wikipedia.org/wiki/Information_retrieval),
tend to begin with known similarities and try to infer associations from
those. Here I'm looking at a kind of reverse situation, where you know
that certain objects are associated, and you don't know anything else
about the objects, but you want to predict whether some other pair of
objects might have an hiterto unknown association.

A classic example in collborative filtering would be: you have a set of
users and a set of products. You know that some users like some products
(and dislike other products). You want to predict from this whether a
particular user would like/dislike a particular product for which they
haven't yet expressed a preference.

This investigation grew out of a personal, now-shelved social website
project. A little foray into linear algebra, statisics and machine
learning - subjects with which I have only little tangental experience - 
turned into a interesting intellectual rabbit hole. I've decided to
try to document my thought processes and findings here, in case it could
be of use or to others.

I've begun laying out the idea in a Jupyter notebook [here](notebooks/vecassoc.ipynb).
The original, rather sprawling and inefficient code from the defunct
website is [here](pickspace).

I have been hunting around for prior art on this sort of approach without
much success, but my day job is not in this field and there is likely
a lot of relevant published research with which I am unfamiliar. If you
know of any, I'd love to [hear about it](mailto:sagland@gmail.com?subject=vecassoc)!
