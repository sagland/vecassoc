# PickProfile
'''
PickSpace is an imaginary world where similar concepts group together, and
unrelated ones drift apart. Concepts can be users, options, contexts, choices,
tags etc. Users express a preference or other association with these other
entities through what they pick (or what they don't pick, and why).

Each of these events causes a set of representative points in PickSpace to be
nudged closer together, or further apart: converging on a configuration
where users are "closer" to other entities they relate to, and closer to 
other users with similar preferences. And they'll be "far" from concepts
they shun. This includes other entities (users, tags, options) that they
haven't actually encountered before, but which other users with similar
preferences have liked. 

PickSpace is the surface of high-dimensional unit hypersphere. This keeps the
system under control: points don't go flying off into distant space; they
can only go as far as the opposite side of the sphere. But the high
dimensionality means there's a lot of "room to move".

Each point on the hypersphere not only has a position but a weight. This
weight represents how "fine-tuned" the position of the point is. Or in other
words, how many "pick" events have contributed to its position.

Each entity begins with a random position on the sphere.

When a pick event indicates that two objects should be "closer", that event has
a certain "weight". This is what happens:
  * A point is chosen on the sphere on the shortest path between the two
    objects.
  * The position is biased toward the object with the higher weight:
      * If the objects have equal weight, the point is right in the middle.
      * If one object has twice the weight, the point is twice as far from
        the other point. 
      * If one point has zero weight (a newbie), and other nonzero, the new 
        point is directly on the nonzero point.
      * and so on. 
  * The two entities have their own weights incremented by the event weight.
  * The two entities' points are rotated toward the choice point, but they
    only move part of the way there (event.weight / entity.weight).
    
This gradual slowing of each move ensures that, over time, an entity's
position is the weighted average of the positions of all the entities
they were rotated toward.

There are also repulsions, where two entities are nudged AWAY from eachother.
On these occasions, the process is mostly the same, but the points the
entities that the entities rotate toward are  on opposite sides of the sphere,
but, again, biased away from the "weightier" entity.

The reason for biasing to the higher weighted object is to respect the fact
that, through experience, they have "earned" their position on the sphere,
whereas new-comers need more freedom to move around and find their place,
without disrupting the established residents too much. 
'''

from math import sqrt, acos, pi, sin, cos, modf, degrees, log
from random import gauss, random, seed, choice, randint, shuffle
from base64 import urlsafe_b64encode, urlsafe_b64decode
from pprint import pprint
from struct import pack, unpack, calcsize
from itertools import imap, izip
import operator
import unittest
import logging

from ext.mpmath import betainc, gamma, inverse
from ext.matlib import matiden, matdiag, matsub, matprod, transpose, \
    matsetblock, matdim, matinverse, matvec

from vecblend import vecblend, random_unit_vector

DIM = 24 # The number of dimensions
TWODIM = 2 * DIM
SIMILARITY_LUT_SIZE = 200

# Pairs of "k" and "L" parameters for the spherical location-sensitive
# hashing. Ordered from least precise to most precise (so that more precise
# searches can be added in future if needed).
# Refs:
# 1. Terasawa, Tanaka (2007) "Spherical LSH for approximate nearest neighbour
# search on unit hypersphere." Procedings of the Workshop on Algorithms and
# Data Structures.
# 2. Andoni, A., Indyk, P.: Near-Optimal Hashing Algorithms for Approximate
# Nearest Neighbor in High Dimensions. In: FOCS'06. Proc. 47th Annual IEEE
# Symposium on Foundations of Computer Science, pp. 459-468 (2006)
MAX_SLSH_K = 8  # Maximum hash length
MAX_SLSH_L = 50 # Maximum number of tables per parameter set
MAX_SLSH_PARAMS = 10 # Maximum pairs of parameters
# Ensure all unique hashes can fit in a 64-bit integer (for neatness).
assert(pow(TWODIM, MAX_SLSH_K) * MAX_SLSH_L * MAX_SLSH_PARAMS < pow(2, 63))
DEFAULT_SLSH_PARAMS = [(2, 5), (3, 10), (4, 15)]

def mix(a, b, blend):
    return b * blend + a * (1 - blend)

def factorial(n):
    f = 1
    while n > 0:
        f = f * n
        n = n - 1
    return f

def double_factorial(n):
    f = 1
    while n > 0:
        f = f * n
        n = n - 2
    return f

def float_to_ubyte(f):
    '''Given a float in the range -1 to 1, returns a quantized signed byte
    in the range 0 - 255.'''
    return int(min(1.0, max(0.0, (f + 1.0) / 2.0)) * 255.0 + 0.5)

def ubyte_to_float(b):
    '''Given, a quantized signed byte in the range 0 - 255, returns a
     float in the range -1 to 1.'''
    return (float(b) / 255.0) * 2.0 - 1.0

def float_to_uint(f):
    '''Given a float >= 0, returns a quantized unsigned integer in the range
    in the range 0 - 4294967295.'''
    return min(4294967295, max(0, int(f)))

def uint_to_float(i):
    '''Given, a unsigned byte in the range 0 - 4294967295. returns a float.'''
    return float(i)

# The arguments below are a recommended optimisation from
# the itertools documentation.
def dot(a, b, sum=sum, imap=imap, mul=operator.mul):
    return sum(imap(mul, a, b))

def angle(a, b, sum=sum, imap=imap, mul=operator.mul):
    try:
        return acos(sum(imap(mul, a, b)))
    except ValueError:
        return acos(min(1.0, max(-1.0, sum(imap(mul, a, b)))))

def compress(l, f):
    '''Compresses a list of numbers l by a factor of int f, by averaging every
    f elements into a single element in the resulting generator.'''
    f = int(f)
    oe = 0.
    for i, e in enumerate(l):
        if i and not (i % f):
            yield oe / f
            oe = 0.
        oe += e
    yield oe / (i % f + 1)

def hypersphere_surface_area(dim=DIM):
    '''From:http://mathworld.wolfram.com/Hypersphere.html'''
    if dim % 2: # Odd dimensions
        return (pow(2.0, (dim + 1) * 0.5) * pow(pi, (dim - 1) * 0.5) /
                double_factorial(dim - 2))
    else: # Even dimension
        return (2 * pow(pi, dim * 0.5) /
                factorial(dim * 0.5 - 1))

def hypersphere_cap_area(angle, proportion_only=False, dim=DIM):
    '''
    This is actually an interesting measure of the "distance" between two
    points on the surface of a hypersphere (given their angle). It represents
    the proportion of surface area (or uniformly distributed points on it)
    that is closer to p1 than p2.
    From: http://scialert.net/fulltext/?doi=ajms.2011.66.70&org=11'''
    assert(0 <= angle <= pi)
    bigcap = angle > pi / 2
    if bigcap: # Measure the smaller cap and subtract it
        angle = pi - angle
    result = float(0.5 * betainc((dim - 1) * 0.5, 0.5, x1=0,
                                 x2=pow(sin(angle), 2.0), regularized=True))
    if bigcap:
        result = 1 - result
    if not proportion_only:
        result *= hypersphere_surface_area(dim)
    return result

def random_rotation_matrix(dim=DIM):
    '''Generates a random rotation matrix.
    See box "How to make a random rotation matrix" in:
    Terasawa, Tanaka (2007) "Spherical LSH for approximate nearest
    neighbour search on unit hypersphere." Procedings of the Workshop on
    Algorithms and Data Structures. pp. 37
    '''
    result = []
    for i in xrange(dim):
        # Generate a random vector on the unit sphere
        vi = random_unit_vector(dim)
        for j in xrange(i):
            vj = result[j]
            dp = dot(vi, vj)
            vi = [a - b * dp for a, b in izip(vi, vj)]
        # Normalize and add the row
        vl = sqrt(sum(k * k for k in vi))
        result.append([k / vl for k in vi])
    return result

def closest_orthoplex_vertex(p, A, dim=DIM):
    '''
    Given a point p on the dim-unit sphere, find the nearest vertex on a
    orthoplex filling the sphere and rotated with the supplied rotation
    matrix A. This is the basis of the locality-sensitive hashing. And it
    seems to be an extremely clever algorithm.
    Ref: Terasawa, Tanaka (2007) "Spherical LSH for approximate nearest
    neighbour search on unit hypersphere." Proceedings of the Workshop on
    Algorithms and Data Structures. pp. 37, fig. 3
    '''
    maxdot, maxv, maxi = -1, -1, -1
    for i in xrange(dim):
        v = [row[i] for row in A] # Column
        absdot = abs(dot(p, v))
        if absdot > maxdot:
            maxdot = absdot
            maxv = v
            maxi = i
    if dot(p, maxv) < 0:
        maxi += dim
    return maxi

#A look up table for the point "similarity" function, which isn't cheap.
__similarity_lut = None
def __build_similarity_lut():
    lut = [-1.0]
    for i in xrange(1, SIMILARITY_LUT_SIZE):
        dot = (2.0 * i / SIMILARITY_LUT_SIZE) - 1
        lut.append(1 - hypersphere_cap_area(acos(dot), True) * 2)
    lut.append(1.0)
    global __similarity_lut
    __similarity_lut = lut
    return lut

def similarity_lut():
    global __similarity_lut
    return __similarity_lut or __build_similarity_lut()

def lookup_similarity_lut(dot):
    '''Given a dot product between two PickPoints, returns the similarity
    as interpolated out of a pre-calculated lookup table.'''
    if dot < -.999999:
        return  (-1.0)
    elif dot > .999999:
        return 1.0
    lut = similarity_lut()
    remain, index = modf(((dot + 1) / 2) * SIMILARITY_LUT_SIZE)
    index = int(index)
    f, t = lut[index], lut[index + 1]
    return f * (1 - remain) + t * remain


class PickPoint(object):
    '''A point on the surface of an n-dimensional unit hypersphere. Data is
    processed in two spaces: cartesian space, and "surface space".
    On the surface of the hypersphere, distance of 1.0 is the "maximum"
    distance between two points: measuring around the sphere to the point on
    the exact opposite side.
    
    That is, the "sdistance" between two points is measured by:
    
    acos(dot(p1, p2)) / pi
    
    Where p1 and p2 are the position vectors of the two points. 
    
    The information stored includes:
    
     * The "position" as an n-dimensional vector in cartesian space.
     * A "weight", as a measure of how finely-tuned this position
       by the accumulated weight of pick results. This is only meaningful
       as it relates to the weight of other points.
     * A "predictability. This is the mean dot product with other
       points to which this point is associated. E.g. a user's point might be
       associated with the point of an option they picked. A predictability
       of 0 indicates randomness. Predictabilities greater than 0 mean that the
       PickPoint's position can predict what a user will pick, with a maximum
       of 1 meaning 100% predictability (impossible). Values less than one
       suggest a trend-bucking point. Not just unusual (low predictability)
       but active inconsistency with the majority of other points.
    '''
    # All floats
    __BYTESTRING_FMT = str(DIM + 2) + 'f'
    # Signed bytes (weight as unsigned short)
    __CBYTESTRING_FMT = str(DIM) + 'BHB'
    __hash_functions = None
    __hash_parameters = None

    def __init__(self, position=None, weight=0.0, predictability=0.0):
        if position is None:
            self.position = self._random_position()
        elif isinstance(position, str):
            p = PickPoint.from_bytestring(position)
            self.position = p.position
            weight = p.weight
            predictability = p.predictability
        else:
            self.position = position
        self.weight = float(weight)
        self.predictability = float(predictability)

    @classmethod
    def dimensions(cls):
        return DIM

    @classmethod
    def bitlength(cls):
        return cls.dimensions() * 64 # Floats

    @property
    def to_dict(self):
        '''Returns a dictionary with the position, weight and predictability
        as the elements.'''
        return {'position': self.position,
                'weight': self.weight,
                'predictability': self.predictability}

    @property
    def to_list(self):
        '''Returns a list of floats containing all the information needed
        to reconstruct this PickPoint. I.e. the first DIM elements are the
        elements of the vector, followed by one element for the weight
        and another for predictability. Essentially keeps it accurate to'''
        return self.position + [self.weight, self.predictability]

    @classmethod
    def from_list(cls, list_):
        return cls(list(list_[0:DIM]), list_[-2], list_[-1])

    @staticmethod
    def cweight(weight):
        '''Compresses a weight value into a 16-bit unsigned integer. Low weight
        values (< 1000) are quite precise while larger values are much rougher
        since the exact weight is not important at these levels. Compressed
        weights are sent to clients but not used for backend calculations.'''
        shift = int(log(max(1, weight), 2.0)) + 1
        # 6 bits for exponent = 64 shift values. Range from -7 to 48.
        shift = min(56, max(-7, shift))
        # 10 bits for fraction
        sweight = int(weight / pow(2, shift - 10))
        return sweight + ((shift + 7) << 10)

    @staticmethod
    def dcweight(cweight):
        '''Decompresses a weight value from a 16-bit unsigned integer to a float.'''
        shift = (cweight >> 10) - 7
        sweight = cweight & 1023
        return float(sweight) * pow(2, shift - 10)

    @property
    def to_clist(self):
        '''Returns data as a list, with position and predictability data
        compressed to unsigned bytes.'''
        return ([float_to_ubyte(e) for e in self.position] +
                [self.cweight(self.weight),
                 float_to_ubyte(self.predictability)])

    @classmethod
    def from_clist(cls, list_):
        return cls(list(ubyte_to_float(e) for e in list_[0:DIM]),
                   cls.dcweight(list_[-2]), ubyte_to_float(list_[-1]))

    @property
    def to_bytestring(self):
        '''Returns the point as a bytestring for storage.'''
        return pack(self.__BYTESTRING_FMT, *self.to_list)

    @classmethod
    def from_bytestring(cls, string):
        '''Recreates a new PickPoint from a bytestring.'''
        try:
            return cls.from_list(unpack(cls.__BYTESTRING_FMT, string)).normalize()
        except: # Bad string?
            logging.warning("Couldn't unpack %s from bytestring.",
                            cls.__name__)
            return None

    @property
    def to_cbytestring(self):
        '''Returns the point as a compressed bytestring.'''
        return pack(self.__CBYTESTRING_FMT, *self.to_clist)

    @classmethod
    def from_cbytestring(cls, string):
        '''Recreates a new PickPoint from a compressed bytestring.'''
        return cls.from_clist(unpack(cls.__CBYTESTRING_FMT, string))

    @classmethod
    def bytesting_size(cls):
        return calcsize(cls.__BYTESTRING_FMT)

    @property
    def to_cb64(self):
        '''Returns the point as a compressed url-safe base64 string (using
        the - and _ characters as well as alphanumeric. The compression means
        the position and predictability data are stored as signed bytes
        instead of floats.'''
        return urlsafe_b64encode(self.to_cbytestring)

    @classmethod
    def from_cb64(cls, string):
        '''Recreates a new PickPoint from a url-safe base64-encoded string.'''
        return cls.from_cbytestring(urlsafe_b64decode(string)).normalize()

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.position,
                                   self.weight, self.predictability)

    def __eq__(self, other):
        try:
            return (all(i == j for i, j in izip(self.position, other.position))
                    and self.weight == other.weight
                    and self.predictability == other.predictability)
        except:
            return False

    @classmethod
    def posblend(cls, p, q, w):
        return vecblend(p, q, w)

    def __add__(self, other):
        '''Adds two PickPoints together, blending the position vector and
        the predictability by the respective weights.'''
        try:
            if not other.weight:
                return self.copy()
            if not self.weight:
                return other.copy()
            weight = self.weight + other.weight
            position = self.posblend(self.position, other.position,
                                     other.weight / weight)
            predictability = mix(self.predictability, other.predictability,
                                 other.weight / weight)
        except AttributeError:
            raise TypeError("PickPoints can only be added to other PickPoints.")
        return self.__class__(position, weight, predictability)

    def __radd__(self, other):
        '''Handles the case of adding a PickPoint to None.'''
        if other is None:
            return self.copy()
        else:
            return other.__add__(self)

    def __sub__(self, other):
        '''Subtracts one PickPoint from another in such a way that if
        p + q = r. Then r - q ~= p. This is useful for "undoing" previous
        additions.'''
        try:
            if not other.weight:
                return self.copy()
            if not self.weight:
                return (-other)
            assert(other.weight <= self.weight)
            weight = self.weight - other.weight
            position = self.posblend(self.position, other.position,
                                     (-other.weight / weight))
            predictability = mix(self.predictability, other.predictability,
                                 (-other.weight / weight))
        except AttributeError:
            raise TypeError("PickPoints can only be subtracted from "
                            "other PickPoints.")
        return self.__class__(position, weight, predictability)

    def __mul__(self, scalar):
        '''Multiple the point by a certain weight scalar.'''
        try:
            scalar = scalar.weight
        except:
            pass # Mustn't be a point
        return self.__class__(self.position, self.weight * scalar,
                              self.predictability)

    def __div__(self, scalar):
        '''Divide the point by a certain weight scalar.'''
        try:
            scalar = scalar.weight
        except:
            pass # Mustn't be a point
        return self.__class__(self.position, self.weight / scalar,
                              self.predictability)

    def __pow__(self, power):
        '''Raise the weight to a certain power.'''
        return self.__class__(self.position, self.weight ** power,
                              self.predictability)

    def __neg__(self):
        '''Return the opposite point.'''
        return PickPoint([-i for i in self.position], self.weight,
                         self.predictability)

    def copy(self):
        '''Returns a copy of this instance.'''
        return self.__class__(self.position, self.weight, self.predictability)

    @staticmethod
    def _random_position():
        '''Generates a random position, uniformly distributed on the
        unit hypersphere.'''
        return random_unit_vector(DIM)

    @property
    def length(self):
        return sqrt(sum(i * i for i in self.position))

    def normalize(self):
        length = self.length
        try:
            self.position = [i / length for i in self.position]
        except ZeroDivisionError:
            self.position = [1.0] + [0.0] * (DIM - 1)
        return self

    def dot(self, other):
        '''The dot product of the position vectors.'''
        return dot(self.position, other.position)

    def angle(self, other):
        '''The angle between this point and another.'''
        dp = self.dot(other)
        if dp <= -1.0:
            return pi
        elif dp >= 1.0:
            return 0
        return acos(dp)

    def distance_sq(self, other):
        '''The cartesian distance to another point (squared).'''
        return sum(i * i for i in
                   (j - k for j, k in izip(self.position, other.position)))

    def distance(self, other):
        '''The cartesian distance to another point.'''
        return sqrt(self.distance_sq(other))

    def sdistance(self, other):
        '''The distance around the surface of the hypersphere to the other
        point. Normalized so that the maximum such distance is 1.'''
        return self.angle(other) / pi

    def similarity(self, other):
        '''Like the sdistance, but normalized so that the distribution of
        uniformly distributed points on the surface is even along the length
        of the distance to the other side of the sphere. This represents the
        proportion of surface area on the hypersphere that's closer to the
        first point than the second point. Another way to put it is this:
        The probability that "other" is closer to "self" than another
        uniformly random point on the sphere (1 being 100% probability).'''
        return 1 - hypersphere_cap_area(self.angle(other), True) * 2

    def similarity_fast(self, other):
        '''Like similarity, but uses a interpolated cached LUT, since the
        usual similarity calculation is slow.'''
        return lookup_similarity_lut(self.dot(other))

    def cmp(self, p1, p2):
        '''Compares two points and ranks them on their similarity to this
        point.'''
        return cmp(self.dot(p1), self.dot(p2))

    @classmethod
    def attract(cls, p1, p2, weight1=1.0, weight2=None):
        '''Given two PickPoints, returns a point directly in between them,
        weight towards the input point with the highest weight. Returns
        two copies of the point, as a delta for each of the input points.
        The returned points have a weight of the input weight, and a
        predictability equal to the similarity between the input points.'''
        if weight2 is None:
            weight2 = weight1
        try:
            blend = p2.weight / (p1.weight + p2.weight)
            # Experimental
            #blend = ((p2.weight + weight2) /
            #         (p1.weight + weight1 + p2.weight + weight2))
        except ZeroDivisionError:
            blend = 0.5
        if blend <= 0.0: # Only the second point affected
            r1 = None
            r2 = cls(p1.position, weight2, 0.0) # Neutral predictability
        elif blend >= 1.0: # Only the first point affected
            r1 = cls(p2.position, weight1, 0.0)
            r2 = None
        else:
            predictability = cls.similarity_fast(p1, p2)
            position = cls.posblend(p1.position, p2.position, blend)
            r1 = cls(position, weight1, predictability)
            r2 = cls(position, weight2, predictability)
        return r1, r2

    @classmethod
    def repel(cls, p1, p2, weight1=1.0, weight2=None):
        '''Given two PickPoints, returns two point directly opposite eachother,
        weighed towards the input point with the highest weight. The points
        are deltas for each of the respective input points.
        The returned points have a weight of the input weight, and a
        predictability equal to the inverse similarity between the input
        points (since the points should have been far apart).'''
        if weight2 is None:
            weight2 = weight1
        try:
            blend = p2.weight / (p1.weight + p2.weight)
            # Experimental
            #blend = ((p2.weight + weight2) /
            #         (p1.weight + weight1 + p2.weight + weight2))
        except ZeroDivisionError:
            blend = 0.5
        if blend <= 0.0: # Only the second point affected
            r1 = None
            r2 = cls((-p1).position, weight2, 0.0) # Neutral predictability
        elif blend >= 1.0: # Only the first point affected
            r1 = cls((-p2).position, weight1, 0.0)
            r2 = None
        else:
            position1 = cls.posblend(p1.position, (-p2).position, blend)
            position2 = cls.posblend(p2.position, (-p1).position, 1.0 - blend)
            predictability = -cls.similarity_fast(p1, p2)
            r1 = cls(position1, weight1, predictability)
            r2 = cls(position2, weight2, predictability)
        return r1, r2

    @classmethod
    def sum(cls, points, predictor=None):
        '''Given a list of PickPoints, returns a PickPoint with the weighted
        average position and predictability and the total weight. Uses vector
        rotation (slow but more accurate).'''
        if not points:
            return cls() # A random weightless point
        position = [1.0] + [0.0] * (DIM - 1)
        weight = 0.0
        predictability = 0.0
        for point in points:
            pweight = point.weight
            weight += pweight
            position = cls.posblend(position, point.position, pweight / weight)
            if predictor is None:
                predictability += point.predictability * pweight
            else:
                predictability += point.similarity(predictor) * pweight
        if weight:
            predictability /= weight
        else:
            # No weight, return a random weightless point
            return cls()
        point = cls(position, weight, predictability)
        point.normalize()
        return point

    @classmethod
    def sum2(cls, points):
        '''Given a list of points, returns a point with the weighted average
        position and predictability and the total weight. Uses recursive
        binary partitioning.'''
        l = len(points)
        if not l:
            return cls()
        elif l == 1:
            return points[0]
        elif l == 2:
            return points[0] + points[1]
        else: # Split
            h = l // 2
            return cls.sum2(points[:h]) + cls.sum2(points[h:])

    @classmethod
    def sum_fast(cls, points):
        '''Given a list of PickPoints, returns a PickPointed with the weighted
        average position and predictability and the total weight.'''
        position = [0.0] * DIM
        weight = 0.0
        predictability = 0.0
        for point in points:
            pweight = point.weight
            weight += pweight
            for i in xrange(DIM):
                position[i] += point.position[i] * pweight
            predictability += point.predictability * pweight
        if weight:
            predictability /= weight
        else:
            predictability = 0.0
        point = PickPoint(position, weight, predictability)
        point.normalize()
        return point

    @classmethod
    def set_hash_parameters(cls, parameters):
        '''A list of 2-tuples, each containing a value of "k" (the number of
        subhashes) and "L" (the number of hash tables) as defined in the LSH
        literature. These sets of parameters should be ordered from least
        precise to more precise. That is the first set should result in the
        most matches, and the last, the least matches. If more precision is
        needed, an extra pair can be appended to a previous set, and it will
        produce the same hashes, with some extra more precises ones. The
        hashes are integers, such that lower ints give more precise results.'''
        assert all(isinstance(i, int) and isinstance(j, int)
                   for i, j in parameters)
        cls.__hash_parameters = parameters
        cls.__hash_functions = None

    @classmethod
    def hash_parameters(cls):
        return (cls.__hash_parameters or DEFAULT_SLSH_PARAMS)

    @classmethod
    def __generate_hash_functions(cls):
        '''Initializes the hash functions.'''
        params = cls.hash_parameters()
        hfs = []
        for k, L in params:
            seed(L * MAX_SLSH_K + k)
            hfs.append([random_rotation_matrix() for _ in xrange(k * L)])
        cls.__hash_functions = hfs
        seed()
        return cls.__hash_functions

    @classmethod
    def hash_functions(cls):
        return cls.__hash_functions or cls.__generate_hash_functions()

    def get_slshes(self):
        '''Returns a list spherical locality-sensitive hashes. Hashes are
        sorted in numerical order, which is from most precise to least precise,
        assuming the parameters have been provided in the opposite order.'''
        result = []
        params = self.hash_parameters()
        # What's the highest value a single hash could have?
        max_hash = pow(TWODIM, MAX_SLSH_K) # Not counting data for table and param-set
        # Give each table a range
        max_table_hash = max_hash * MAX_SLSH_L
        hfs = self.hash_functions()
        for s, (k, L) in enumerate(params):
            for i in xrange(L):
                slsh = 0 # Let's accumulate them into an int.
                for j in xrange(k):
                    mtx = hfs[s][i * k + j]
                    subhash = closest_orthoplex_vertex(self.position, mtx)
                    # Each subhash is an int >= 0 and < DIM * 2.
                    slsh += subhash * pow(TWODIM, j)
                # Unique range for each hash table
                slsh += i * max_hash
                # Unique range for each param pair, in REVERSE order, so that
                # the most precise is lowest, and sorted first.
                slsh += (MAX_SLSH_PARAMS - s) * max_table_hash
                result.append(slsh)
        result.sort()
        return result

    @classmethod
    def __generate_axes3d(cls, dim=DIM, maxiter=20000):
        '''Returns a list of DIM 3D vectors evenly spaced over the unit
        sphere.'''
        axes = [random_unit_vector(3) for _ in xrange(dim)]
        for _ in xrange(maxiter):
            mind, mini, minj = 1.0, -1.0, -1.0
            for i, a in enumerate(axes):
                for j, b in enumerate(axes):
                    if i == j: continue
                    dist = 1.0 - abs(dot(a, b))
                    if dist < mind:
                        mind, mini, minj = dist, i, j
            # Shift the nearest points further apart
            if dot(axes[mini], axes[minj]) < 0:
                axes[minj] = [-a for a in axes[minj]]
            axes[minj] = [a + 1.01 * (b - a)
                          for a, b in izip(axes[mini], axes[minj])]
            axes[mini] = [a - 0.01 * (b - a)
                          for a, b in izip(axes[mini], axes[minj])]
            li = sqrt(sum(a * a for a in axes[mini]))
            lj = sqrt(sum(a * a for a in axes[minj]))
            axes[mini] = [a / li for a in axes[mini]]
            axes[minj] = [a / lj for a in axes[minj]]
        return axes

    __axes3d = [[0.21362371296647906, -0.93332828413584878, 0.28855367488640193], [0.78359889130427751, -0.56427016976951916, -0.25994605797162085], [0.66611395073515256, -0.7142586753412048, 0.21477138854101169], [-0.21361864556289711, -0.81305313164269077, 0.54158256932243187], [-0.76107033654174727, -0.098502704435245877, -0.64114675391455755], [-0.69233866492496676, -0.56420541162270954, -0.44982154966782523], [-0.3939874978392322, -0.39825912380452699, -0.82834987888743694], [0.84990495747310102, -0.1279459082355279, -0.51116671236342492], [0.036270147880130825, -0.17502562829458926, -0.9838955766812032], [-0.68610356400102435, -0.72673178325626697, 0.033508426854425961], [-0.26867640499135592, -0.85843116923582485, -0.43693124983838771], [-0.13421058061068078, -0.34974940752962019, 0.92718006448845913], [-0.24120997435975555, -0.96874767309433918, 0.057841975620387773], [0.53322036993840938, -0.48112455000732235, -0.69584136443804301], [-0.33831449421700782, 0.16305550305649361, -0.92679890263513165], [0.24135914630869179, -0.91010415375974862, -0.33683258720377901], [0.94471233355912632, -0.29425198048456297, 0.1446871756659257], [0.50322312695269089, -0.00068763739039668888, -0.86415624261748836], [-0.88851710852172694, -0.34015960384598443, 0.30794283848716608], [0.23234787820998337, -0.64493376760062215, 0.72806242788636288], [0.95763228575299009, 0.22051072606718225, 0.18524423060605083], [0.053632643816083146, -0.63774560106981182, -0.7683775685386719], [-0.66842241289826021, 0.40745625002960256, -0.62224664100907912], [0.56227293524126953, 0.47064409094126614, -0.67995829721920187]]

    @classmethod
    def get_axes3d(cls):
        '''Returns DIM 3D unit vectors evenly spread over the sphere.'''
        return cls.__axes3d

    def position_as_3d(self):
        '''Returns the position compressed to a 3D vector. Each elements is
        placed on a random, but evenly space, 3D axis. Relative proximity of
        compared points should be preserved to an extent.'''
        r = [0.0, 0.0, 0.0]
        for e, axis in izip(self.position, self.get_axes3d()):
            r = [r[0] + axis[0] * e, r[1] + axis[1] * e, r[2] + axis[2] * e]
        l = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        r = [r[0] / l, r[1] / l, r[2] / l]
        return r

class TestPickSpace(unittest.TestCase):

    def setUp(self):
        seed(12345)
        self.testn = 10000
        self.bucketn = 30
        self.dimensions = DIM

    def testRandomVectorUniformity(self):
        buckets = [0 for _ in range(self.bucketn)]
        for test in (PickPoint().similarity_fast(PickPoint())
                     for x in range(self.testn)):
            buckets[int(((test + 1) / 2) * self.bucketn)] += 1
        #print buckets
        #print ['%.1f%%' % (x / float(self.testn) * 100) for x in buckets]
        fact = float(self.testn) / self.bucketn
        assert(not any(abs(bucket - fact) / fact > 0.25 for bucket in buckets))

    def testRandomRotationMatrixUniformity(self):
        buckets = [0] * DIM * 2
        for _ in xrange(self.testn):
            p = [0] * DIM
            p[0] = 1
            #p = random_unit_vector()
            A = random_rotation_matrix()
            hash = closest_orthoplex_vertex(p, A)
            buckets[hash] += 1
        #print buckets
        fact = float(self.testn) / len(buckets)
        assert(not any(abs(bucket - fact) / fact > 0.25 for bucket in buckets))

    def testFastSimilarityLUT(self):
        for _ in xrange(1000):
            p = PickPoint()
            q = PickPoint()
            s = p.similarity(q)
            sf = p.similarity_fast(q)
            assert(abs(s - sf) < 0.05) # Good enough for a percentage

    def testPointBlending(self):
        '''Not so much a test as a record of my fooling around while evaluating
        this technique.'''
        testn = DIM * DIM
        ps = []
        avgsims = [0]
        r = PickPoint()
        total_weight = 0
        for i in xrange(testn):
            p = PickPoint()
            weight = 1.0
            total_weight += weight
            blend = weight / total_weight
            r.position = vecblend(r.position, p.position, blend)
            ps.append(p)
            avgsims.append(mix(avgsims[i], r.similarity_fast(p), blend))
        assert(abs(1 - r.length) < .999)
        avgsims.pop(0)
        print 'Length of r: %f' % r.length
        avgsim = sum(r.similarity_fast(p) for p in ps) / testn
        print ('Average similarity: %f | %s' %
               (avgsim, ','.join('%.2f' % s for s in compress(avgsims, testn / 25))))
        print ','.join('%.1f' % s
                       for s in compress((r.similarity_fast(p) for p in ps), testn / 25))

    def testSphereicalHashingConsistency(self):
        '''Ensures that when tacking on new more-precise hashing parameters that
        the initial hashes remain the same.'''
        points = [PickPoint() for _ in xrange(10)]
        PickPoint.set_hash_parameters([(2, 5)])
        slshes = [p.get_slshes() for p in points]
        PickPoint.set_hash_parameters([(2, 5), (3, 10)])
        new_slshes = [p.get_slshes() for p in points]
        assert(all(slsh == new_slsh[-len(slsh):]
                   for slsh, new_slsh in zip(slshes, new_slshes)))
        PickPoint.set_hash_parameters([(2, 5), (3, 10), (4, 15), (5, 20)])
        newer_slshes = [p.get_slshes() for p in points]
        assert(all(slsh == new_slsh[-len(slsh):]
                   for slsh, new_slsh in zip(new_slshes, newer_slshes)))

    def testPointAdditionAndSubtraction(self):
        for _ in xrange(100):
            p = PickPoint(None, random() * 100, random())
            q = PickPoint(None, random() * p.weight * 0.1, random())
            r = p + q
            s = r - p
            dist = q.sdistance(s)
            predict_loss = abs(q.predictability - s.predictability)
            assert(dist < 1.0e-5 and predict_loss < 1.0e-5)

    def testCB64Encoding(self):
        '''Ensure a minimum of data loss when compressing a PickPoint to a short
        base64-encoded bytestring.'''
        for _ in range(10):
            p = PickPoint()
            p.weight = random() * 10000.0
            p.predictability = random() * 2 - 1
            p64 = p.to_cb64
            q = PickPoint.from_cb64(p64)
            assert(p.sdistance(q) < 0.01)


def sphericalHashingSearchTest():
    n = 100
    ntests = 20
    params = [(2, 5), (3, 10), (4, 15)]
    def search(q, limit=None):
        i = 0
        for slsh in q.get_slshes():
            try:
                for p in hash_tables[slsh]:
                    yield p
                    i += 1
                    if limit is not None and i >= limit:
                        return
            except KeyError:
                continue
    hash_tables = {}
    PickPoint.set_hash_parameters(params)
    for _ in xrange(n):
        p = PickPoint()
        for slsh in p.get_slshes():
            try:
                hash_tables[slsh].append(p)
            except KeyError:
                hash_tables[slsh] = [p]
    for _ in  xrange(ntests):
        q = PickPoint()
        print sorted((q.similarity_fast(p) for p in search(q, 10)),
                     reverse=True)


def sphericalHashingAnalysis():
    def analyse(length, table_count, testn=1000, bucketn=50):
        PickPoint.set_hash_parameters([(length, table_count)])
        points = []
        for _ in xrange(testn):
            p = PickPoint()
            slshes = p.get_slshes()
            points.append((p, slshes))

        hits = [0] * bucketn
        misses = [0] * bucketn
        for _ in xrange(testn):
            q = PickPoint()
            qslshes = q.get_slshes()
            for p, pslshes in points:
                dist = 2 - (q.similarity_fast(p) / 2 + .5)
                #dist = q.sdistance(p)
                bucket = int(dist * bucketn)
                if any(i == j for i, j in zip(pslshes, qslshes)):
                    hits[bucket] += 1
                else:
                    misses[bucket] += 1
        results = [float(hits[i]) / ((hits[i] + misses[i]) or 1)
                   for i in xrange(bucketn)]
        hitsn, missesn = sum(hits), sum(misses)
        alln = hitsn + missesn
        return results, float(hitsn) / alln

    for length in [1, 2, 3, 4, 5]:
        for table_count in [1, 2, 4, 8, 16, 32]:
            results, proportion = analyse(length, table_count,
                                            testn=1000, bucketn=50)
            line = [length, table_count, proportion] + results
            print ','.join('%r' % d for d in line)

def pointAveragingAnalysis():
    MAXPOINTS = 50
    TESTS = 20
    results = []
    for dim in [2, 4, 8, 16, 24]:
        result = []
        for n in xrange(1, MAXPOINTS + 1):
            avg = 0.00
            for _ in xrange(TESTS):
                points = [random_unit_vector(dim) for _ in xrange(n)]
                # Determine a point as equidistant from each of these points
                # as we can.
                p = points[0]
                for i, q in enumerate(points[1:]):
                    blend = 1.0 / (i + 2)
                    p = vecblend(p, q, blend, 'rotate')
                avg += sum(1 - hypersphere_cap_area(angle(p, q),
                                                    True, dim=dim) *
                               2 for q in points) / n
            result.append((n, avg / TESTS))
        results.append((dim, result))
        print dim
    print 'dimensions, ' + ', '.join('%d' % x for x in xrange(1, MAXPOINTS + 1))
    for dim, result in results:
        print ('%d, ' % dim) + ', '.join('%.4f' % x[1] for x in result)

def testPointAveragingConsistency():
    z = random_unit_vector(DIM)
    points = [vecblend(random_unit_vector(DIM), z, 0.0, 'rotate')
              for _ in xrange(100)]
    avgs = []
    for _ in xrange(10):
        shuffle(points)
        p = [1] + [0] * (DIM - 1)
        for i, q in enumerate(points):
            blend = 1.0 / (i + 1)
            p = vecblend(p, q, blend, 'rotate')
        avgs.append(p)
    avg_zangle = 0.0
    avg_angle = 0.0
    weight = 0.0
    for i, p in enumerate(avgs):
        avg_zangle += angle(z, p)
        for j, q in enumerate(avgs):
            if i == j:
                continue
            avg_angle += angle(p, q)
            weight += 1
    avg_zangle /= len(avgs)
    avg_angle /= weight
    print degrees(avg_angle), degrees(avg_zangle)




if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestPickSpace)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #import json
    #print json.dumps(similarity_lut())
    #pointAveragingAnalysis()
    testPointAveragingConsistency()
