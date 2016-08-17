# PickProject
"""A binary alternative to the existing PickPoint implementation."""
from random import getrandbits, random, randrange
from struct import pack, unpack, calcsize
from math import degrees, factorial, fmod, acos, pi
from ext import mpmath
import logging

from pickspace import PickPoint, float_to_ubyte, mix

# Bits in a pickpoint string
DIM = 1024
assert(not DIM % 8)
FDIM = float(DIM)
TWO_TOTHE_DIM = 1 << DIM
FACTORIAL_DIM = factorial(DIM)

# A vector of all ones.
ONES = sum(1 << i for i in range(DIM))

# A vector of all zeros.
ZEROS = 0

# A pregenerated of vectors in which only one bit is set, in order
SINGLEBITS = [1 << i for i in range(DIM)]

def randpos(dim=DIM):
    '''Generates a random position string.'''
    return getrandbits(dim)

def hamdist(p, q):
    '''Returns the Hamming Distance between two bit strings (represented as
    unsigned ints).'''
    diff = p ^ q
    dist = 0
    while diff:
        dist += 1
        diff = diff & (diff - 1)
    return dist

_DIMRATIOLUT = None
def dimratio(x, deriv=0):
    '''The proportion of the dimensions (of a hamming distnace, say).'''
    if deriv > 1:
        return 0.0
    elif deriv == 1:
        return 1.0
    global _DIMRATIOLUT
    try:
        try:
            return _DIMRATIOLUT[x + DIM]
        except TypeError:
            _DIMRATIOLUT = [(i - DIM - 0) / FDIM for i in range(DIM * 2 + 2)]
            return _DIMRATIOLUT[x + DIM]
    except IndexError:
        raise ValueError("dimratio requires a value between %d and %d: got %s" %
                         (-DIM, DIM, x))

# Interpreting points as positions on a DIM-dimensional hypershere
def dot(hd):
    return dimratio(DIM - hd * 2)

def angle(hd):
    return acos(dot(hd))

def normangle(hd):
    return angle(hd) / pi

def binomial_coefficient(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

def chance_of_k_matches(k):
    numer = FACTORIAL_DIM
    denom = factorial(DIM - k) * factorial(k) * TWO_TOTHE_DIM
    r = mpmath.mpf(numer) / denom
    return r

def chance_of_k_or_fewer_matches(k):
    '''
    http://www.wolframalpha.com/input/?i=sum%28n%21+%2F+%28%28n-k%29%21+*+k%21+*+2%5En%29%2C+k+%3D+0+to+x%29
    '''
    return 1.0 - ((mpmath.hyp2f1(1.0, -DIM + k + 1, k + 2, -1) * FACTORIAL_DIM) /
                  (factorial(k + 1) * factorial(DIM - k - 1) * TWO_TOTHE_DIM))

# This is a 2D lut which each element is a lut being a derivative of the previous
# in practise only 3 should be needed?
_HAMDIST_PROBABILITY_LUT = [None, None, None]
def probability_distance(dist, deriv=0):
    '''Given an hamming distance, returns an alternative metric between
    0 and 1 representing the probability that a random bitstring would be
    closer than the hamming distance supplied. This is simply the inverse of
    "similarity".'''
    global _HAMDIST_PROBABILITY_LUT
    if deriv > 0:
        # Return the approximate "deriv"th-order derivative.
        if dist < 0 or dist > DIM:
            # Out of range, assume flat. (not quite right but hey)
            return 0.0
        try:
            return _HAMDIST_PROBABILITY_LUT[deriv][dist]
        except TypeError:  # Lut needs to be built
            _HAMDIST_PROBABILITY_LUT[deriv] = []
            for i in range(DIM + 1):
                p = probability_distance(i - 1, deriv - 1)
                t = probability_distance(i, deriv - 1)
                a = probability_distance(i + 1, deriv - 1)
                r = ((t - p) + (a - t)) * 0.5 * (DIM + 1)
                # r = (a - p) * (DIM + 1)
                # print deriv, i, p, t, a, r
                _HAMDIST_PROBABILITY_LUT[deriv].append(r)
            return _HAMDIST_PROBABILITY_LUT[deriv][dist]
    # The 0-th order derivative, i.e. the actual LUT.
    if dist < 0:
        return 0.0
    elif dist > DIM:
        return 1.0
    try:
        return _HAMDIST_PROBABILITY_LUT[0][dist]
    except TypeError:  # Lut needs to be built
        mpmath.mp.prec = DIM * 2
        _HAMDIST_PROBABILITY_LUT[0] = []
        pckof = 0.0
        for i in range(DIM):
            ckof = chance_of_k_or_fewer_matches(i)
            ck = ckof - pckof
            dr = dimratio(i)
            # We'd like a nice balanced curve, but the "chance of k or
            # fewer" test biases the probability upwards, while "chance of
            # fewer than k  has the opposite problem. A neat solution seems to
            # be to blend from one to the other in such a way that where the
            # hamming distance  is 0 the result of this function is 0 and when
            # the hamming distance  is DIM (the maximum) the return value is 1.
            # When the hamming distance is exactly DIM / 2, this function
            # returns 0.5.
            # Also, pd((DIM / 2) - n) = pd(1 - ((DIM / 2) + n)).
            _HAMDIST_PROBABILITY_LUT[0].append(ckof - ck * (1.0 - dr))
            pckof = ckof
        _HAMDIST_PROBABILITY_LUT[0].append(mpmath.mpf(1.0))
        return _HAMDIST_PROBABILITY_LUT[0][dist]
    except IndexError:
        raise ValueError("probability_distance requires a value between %d and %d: got %s" %
                         (0, DIM, dist))

def similarity(dist):
    '''The similarity of two bitstrings is the reverse (1-x) of their probability
    distance.'''
    return probability_distance(DIM - dist)

def hamdist_reduce(hd, b, f=normangle):
    '''Given a hamming distance, returns the number of bit flips require to
    reduce the distance by b. (b = 0 means no change, b = 1 means all different
    bits are flipped). f is an optional function which converts the hamming
    distance into an alternative normalized metric. f should be a fast: a
    LUT, really.'''
    # The distance metric
    dist = f(hd)
    # The goal distance
    goal = dist * (1.0 - b)
    # Binary search to find the the integer result.
    lo, hi = 0, hd - 1
    while True:
        mid = (lo + hi) // 2
        fmid, fmid1 = f(mid), f(mid + 1)
        if fmid1 < goal:
            lo = mid + 1
        elif fmid >= goal:
            hi = mid
        else:
            break
    # Linear interpolation
    diff = mid + (goal - fmid) / (fmid1 / fmid)
    return hd - diff


def binblend(p, q, b):
    '''
    Blends between two DIM - length binary vectors represented as long ints.

    Returns a new vector that has a "distance" dp to p and dq to q such that:
        dp / (dp + dq) == b
    - with a precision depending on the size of DIM and the distance between
    p and q. Precision errors should be properly distributed to eliminate bias.

    What defines "distance" best is still to be determined at the time of
    writing. We'll start with the hamming distance but I suspect "similarity"
    may be a more useful metric since it represents the probability a randomly
    generated binary vector being at least that close. 
    '''
    # Take care of trivial results
    if b <= 0:
        return p
    elif b >= 1:
        return q
    # Depending on the the blend factor, start with p or q.
    elif b <= 0.5:
        r = p
    else:
        r = q
        b = 1.0 - b

    # Which bits differ?
    diff = p ^ q

    # Test for identity
    if not diff:  # p == q
        return p

    # How many bits differ? (hd = hamming distance)
    hd, rdiff = 0, diff
    while rdiff:
        hd += 1
        rdiff = rdiff & (rdiff - 1)

    # Determine the normalised "distance" where
    # 0: p == q
    # 1: p == q ^ ONES (all bits different)
    # The curve in between is determined by the chosen distance metric.
    # Then determine how many bit would need to be flipped (including partial
    # bits) to reduce the distance by b.

    # Disabling this for now, may have been barking up the wrong tree?
    # fnflips = hamdist_reduce(hd, b)

    # Normalized hamming distance.
    dist = dimratio(hd)
    # How many bits should we flip?
    fnflips = dist * b * DIM

    # Probablistically round to the nearest integer number of flips
    nflips = int(fnflips) + int(random() < fmod(fnflips, 1.0))
    # Why not just round? Because there's only a small finite number of
    # distances between binary strings (=DIM), so there's a chance of systematic
    # bias.

    # Are we even flipping anything?
    if not nflips:
        return r

    # The diff variable contains which bits (1s) would need to be flipped to
    # make p into q, or q into p. We only want nflips of them. An unbiased
    # selection of them.
    # Then we just have to that with xor r and we're done.

    # Find the differing bits, and flip some of them.
    # Wind down hd and nflips as we go to keep track of what we've done.
    for bit in (bit for bit in SINGLEBITS if bit & diff):  # For each differing bit
        # Chance of remaining bits needing flipping.
        if nflips and randrange(hd) < nflips:
            # This will be flipped
            # One fewer flips to do
            nflips -= 1
        else:
            # This bit won't be flipped; remove it from the diff.
            diff ^= bit
        # One fewer different bits left to process
        hd -= 1
    assert(nflips == 0 and hd == 0)
    # Flip the remaining bits on r
    r ^= diff
    return r

class PickPoint2(PickPoint):
    '''A string of n bits representing a position in n-dimensional space.
    Of course there can only be two values in each dimension. The could be
    thought of as being either -1 or 1. In this case the string can be
    considered and n-dimensional vector that, if normalized, would be a point
    on an  n-dimensional unit hyperphere.
    
    The idea is to take the existing PickPoint imlementation (and the time of
    writing) to extremes: maximal dimensionality. It also reduces a lot of
    the operations involved in pickspace comparision and manipulation to
    efficient binary operations.'''

    # All floats
    __BYTESTRING_FMT = str(DIM / 8) + 'B2f'  # TODO
    __CBYTESTRING_FMT = str(DIM / 8) + 'BHB'

    def __init__(self, position=None, weight=0.0, predictability=0.0):
        if position is None:
            self.position = self._random_position()
        else:
            self.position = position
        self.weight = float(weight)
        self.predictability = float(predictability)

    @classmethod
    def dimensions(cls):
        return DIM

    @classmethod
    def bitlength(cls):
        return cls.dimensions()

    @staticmethod
    def _random_position():
        '''Generates a random position string.'''
        return randpos(DIM)

    def _position_to_ubytes(self):
        return [int(255 & self.position >> (8 * i)) for i in range(DIM / 8)]

    @property
    def to_list(self):
        '''Returns a list of floats containing all the information needed
        to reconstruct this PickPoint. I.e. the first DIM elements are the
        elements of the vector, followed by one element for the weight
        and another for predictability. Essentially keeps it accurate to'''
        return self._position_to_ubytes() + [self.weight, self.predictability]

    @property
    def to_clist(self):
        '''Returns data as a list, with position and predictability data
        compressed to unsigned bytes.'''
        return (self._position_to_ubytes() +
                [self.cweight(self.weight),
                 float_to_ubyte(self.predictability)])

    @classmethod
    def from_clist(cls, list_):
        raise NotImplementedError

    @property
    def to_bytestring(self):
        '''Returns the point as a bytestring for storage.'''
        return pack(self.__BYTESTRING_FMT, *self.to_list)

    @classmethod
    def from_bytestring(cls, string):
        '''Recreates a new PickPoint from a bytestring.'''
        try:
            return cls.from_list(unpack(cls.__BYTESTRING_FMT, string)).normalize()
        except:  # Bad string?
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
    def bytestring_size(cls):
        return calcsize(cls.__BYTESTRING_FMT)

    def __eq__(self, other):
        try:
            return (self.position == other.position
                    and self.weight == other.weight
                    and self.predictability == other.predictability)
        except:
            return False

    def __neg__(self):
        '''Return the opposite point.'''
        return PickPoint2(self.position ^ ONES,
                          self.weight, self.predictability)

    def copy(self):
        '''Returns a copy of this instance.'''
        return PickPoint2(self.position, self.weight, self.predictability)

    @classmethod
    def posblend(cls, p, q, w):
        return binblend(p, q, w)

    def hamdist(self, other):
        '''Returns the Hamming distance between this point and another.'''
        return hamdist(self.position, other.position)

    def distance_sq(self, other):
        '''Assumes position bitstring represents a vector where 0 maps to -1
        and 1 maps to 1. Returns the cartesian distance.'''
        # For unmatched bits: 1 - -1 == 2. 2 * 2 = 4
        return self.hamdist(other) * 4

    def normalize(self):
        '''N/A'''
        pass

    def dot(self, other):
        '''Assumes position bitstring represents a vector where 0 maps to -1
        and 1 maps to 1 AND the vector is normalized. Returns the dot product.'''
        # Matched bits: 1, unmatched: -1. Ends up being DIM - hamdist * 2
        # prior to normalization
        undot = DIM - self.hamdist(other) * 2
        # Now the length of this theoretical vector is sqrt(DIM) (think about it).
        # Since the dot product is the product of the length of both vectors
        # (and the cos of the angle) we can divide by sqr(sqrt(DIM)) = DIM to
        # get the normalized dot product.
        return dimratio(undot)

    def similarity(self, other):
        return float(similarity(self.hamdist(other))) * 2 - 1

    def similarity_fast(self, other):
        return self.similarity(other)

    def cmp(self, p1, p2):
        return -cmp(self.hamdist(p1), self.hamdist(p2))

    @classmethod
    def sum(cls, points, predictor=None):
        '''Given a list of PickPoints, finds their weighted average. An improved
        method on simply repetitive bin-blending.'''
        if not isinstance(points, list):
            points = list(points)  # We're going to be iterating this a lot
        if not points:
            return cls()  # A random weightless point
        weight = sum(p.weight for p in points)
        if not weight:
            return cls()  # No weight, no point adding up points or predictability
        halfweight = weight / 2
        position = sum(SINGLEBITS[i] for i in range(DIM) if
                       int(sum(p.weight for p in points
                               if SINGLEBITS[i] & p.position) / halfweight))
        if predictor is None:
            # Average the predictability of the input points
            predictability = sum(p.predictability * p.weight for p in points)
        else:
            # Generator predictability from the predictor point
            predictability = sum(p.similarity(predictor) * p.weight
                                 for p in points)
        predictability /= weight
        return cls(position, weight, predictability)

r = randpos()
q = randpos()

def test():

    # for i in range(DIM + 1):
    #    pd = probability_distance(i)
    #    s = similarity(DIM - i)
    #    print '%03d: %1.15f  %1.15f  %1.15f' % (i, pd, s, pd - s)
    # print; print
    #

    # for i in range(DIM + 1):
    #    #print '%03d: %1.15f, %s' % (i, dimratio(i), mpmath.nstr(probability_distance(i), 15))
    #    print('%03d: %1.15f, %1.15f, %1.15f, %1.15f' %
    #          (i, dimratio(i), probability_distance(i),
    #           probability_distance(i, 1), probability_distance(i, 2)))

    def hdsm(a, b):
        hd = hamdist(a, b)
        return hd, float(similarity(hd))

    # p = randpos()
    # q = randpos()
    # s = randpos()
    # r = binblend(p, q, 0.5)
    # r = binblend(r, s, 1.0 / 3.0)
    # print hdsm(p, q)
    # print hdsm(p, s)
    # print hdsm(q, s)
    # print
    # print hdsm(p, r)
    # print hdsm(q, r)
    # print hdsm(s, r)
    # print
    # print hdsm(p, binblend(p, p ^ ONES, 0.5))

    ps = [PickPoint2(weight=i) for i in range(1, 1000)]
    r = PickPoint2.sum(ps)
    for p in sorted(ps, r.cmp):
        print '+' * int(p.weight / 10)

    quit()

    for npoints in range(1, 10000, 50):
        ps = [PickPoint2(weight=1.0) for i in range(npoints)]
        r = PickPoint2.sum(ps)
        # print '%03d:' % npoints, sum(dimratio(r.hamdist(p)) for p in ps) / float(npoints)
        print '%03d:' % npoints, sum(r.similarity(p) for p in ps) / float(npoints)

    quit()

    classes = [PickPoint, PickPoint2]
    ntests = 100
    for _class in classes:
        for npoints in range(1, 101):
                avgs = 0.0
                avgds = 0.0
                for t in range(ntests):
                    ps = [_class(weight=1.0) for i in range(npoints)]
                    # r = _class(weight=1.0)
                    r = _class.sum(ps)
                    sims = [r.similarity(p) for p in ps]
                    avg = sum(sims) / npoints
                    avgd = sum(abs(x - avg) for x in sims) / npoints
                    avgs += avg
                    avgds += avgd - (1 - avg)
                avgs /= ntests
                avgds /= ntests
                bl = _class.bitlength()
                print ('"%s(%d)"\t%d\t%d\t%.6f\t%.6f' %
                       (_class.__name__, _class.dimensions(), bl, npoints, avgs, avgds))

    # hamdist_probability(0)
    # from timeit import Timer
    # t = Timer('binblend(r, q, 0.5)',
    #          'from __main__ import randpos, binblend, r, q')
    # print 1.000000 * t.timeit(number=1000) / 1000

    # p1 = PickPoint2().position
    # p2 = PickPoint2().position
    # zero = PickPoint2(ZEROS)
    # s = 0
    # for i in range(DIM + 1):
    #    p = PickPoint2((1 << i) - 1)
    #    os = s
    #    s = p.sdistance(zero)
    #    print p.hamdist(zero), abs(s - os)
    #    #print (p.hamdist(zero), p.dot(zero), a)
    # p = PickPoint2()
    # assert(p.hamdist(-p) == DIM)
    # qs = sorted([PickPoint2() for _ in range(100)], p.cmp)
    # for q in qs:
    #    print p.hamdist(q)
    # print binomial_coefficient(DIM, 192)
    # total = 0.0
    # for i in range(DIM / 2):
    #    i = DIM / 2 - i
    #    p = chance_of_k_matches(i)
    #    total += p
    #    print '%3d: ' % i, p, total
    # print 'Total:', total
    # mpmath.mp.dps = 100
    # mpmath.mp.pretty = True
    # for i in range(DIM + 1):
    #    print '%3d: ' % i, hamdist_probability(i)

    # pcs = [0 for i in range(10)]
    # for i in range(10000):
    #    p1, p2 = PickPoint2(), PickPoint2()
    #    similarity = p1.similarity(p2)
    #    pcs[int(similarity * 10)] += 1
    # for i, n in enumerate(pcs):
    #    print i, n

    # for i in range(100):
    #    p1, p2 = PickPoint2(), PickPoint2()
    #    print degrees(p1.angle(p2))

if __name__ == '__main__':
    test()
