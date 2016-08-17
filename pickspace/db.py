# PickProject
"""Classes for associating pickspace with the datastore."""

import random
#import logging
from itertools import izip
from google.appengine.api.memcache import _CLIENT as memcache
from google.appengine.ext import db

from pickspace import PickPoint

REINDEX_TOLERANCE = 10.0
REINDEX_MIN_WEIGHT = 5.0
REINDEX_MIN_PREDICTABILITY = 0.25

class PickPointProperty(db.BlobProperty):
    '''A datastore property for storing PickPoints.'''

    # Tell what the user type is.
    data_type = PickPoint

    # For writing to datastore.
    def get_value_for_datastore(self, model_instance):
        point = super(PickPointProperty,
                      self).get_value_for_datastore(model_instance)
        return db.Blob(point.to_bytestring) if point else None

    # For reading from datastore.
    def make_value_from_datastore(self, value):
        if not value:
            return None
        return PickPoint.from_bytestring(value)

    def validate(self, value):
        if value is None:
            return None
        if not isinstance(value, PickPoint):
            raise db.BadValueError('Property %s must be a PickPoint (%s)' %
                                (self.name, value))
        return super(PickPointProperty,
                     self).validate(db.Blob(value.to_bytestring))

    def empty(self, value):
        return not value


class PickspaceModel(db.Model):
    '''A base class for models which have a position in pickspace.'''

    pp_reindex_tolerance = REINDEX_TOLERANCE
    pp_reindex_min_weight = REINDEX_MIN_WEIGHT
    pp_reindex_min_predictability = REINDEX_MIN_PREDICTABILITY

    pickpoint = PickPointProperty(required=False, indexed=False)

    @property
    def predictability(self):
        '''Returns the pickspace predictability of this entity. Based on how
        often a pick associates this entity with another entity which already
        has a similar pickpoint.'''
        return self.pickpoint.predictability

    @property
    def pickpoint_as_dict(self):
        '''Returns the pickspace position, weight and predictability as a
        dictionary.'''
        if self.pickpoint:
            return self.pickpoint.to_dict
        else:
            return None

    @property
    def pickpoint_as_cb64(self):
        '''The pickpoint compressed (1 byte per dimension) encoded in
        URL-safe Base64.'''
        if self.pickpoint:
            return self.pickpoint.to_cb64
        else:
            return None

    @classmethod
    def batch_get_pickpoints(cls, keys):
        '''Given a list of PickspaceModel keys, generates tuples of keys
        and corresponding PickPoints efficiently (using cacheing). Not
        necessarily in the same order.'''
        def cachekey(key):
            return '%s/%s' % (key.kind(), key.id_or_name())
        BATCH_SIZE = 500
        uncached_keys = []
        # First return any cached points
        batch, remain = keys[:BATCH_SIZE], keys[BATCH_SIZE:]
        while batch:
            cached = memcache.get_multi([cachekey(key) for key in batch],
                                        key_prefix='PickPoint/')
            # Start yielding the cached results
            for key in batch:
                cache_key = cachekey(key)
                try:
                    point = cached[cache_key]
                    if point is not None:
                        point = PickPoint.from_bytestring(point)
                    yield key, point
                except KeyError:
                    uncached_keys.append(key)
            batch, remain = remain[:BATCH_SIZE], remain[BATCH_SIZE:]
        # Grab the remaining points from the database
        batch, remain = uncached_keys[:BATCH_SIZE], uncached_keys[BATCH_SIZE:]
        while batch:
            for_cache = {}
            for key, entity in zip(batch, db.get(batch)):
                cache_key = cachekey(key)
                try:
                    point = entity.pickpoint
                    yield key, point
                    for_cache[cache_key] = point.to_bytestring
                except: #TODO: should we be more careful here?
                    yield key, None
                    for_cache[cache_key] = None
            if for_cache:
                memcache.set_multi(for_cache, key_prefix='PickPoint/')
            batch, remain = (remain[:BATCH_SIZE]), remain[BATCH_SIZE:]

    @classmethod
    def batch_cache_pickpoints(cls, points):
        '''Given a list of tuples containing entity keys and pickpoints,
        caches these in the memcache.'''
        def cachekey(key):
            return '%s/%s' % (key.kind(), key.id_or_name())
        for_cache = dict((cachekey(point[0]), point[1].to_bytestring)
                          for point in points)
        memcache.set_multi(for_cache, key_prefix='PickPoint/')

    def pickpoint_indexable(self):
        '''Returns True or False as to whether this entity's pickpoint should
        be indexed. Whether or not it is may still depend on other factors.'''
        return True

    def pickpoint_needs_reindexing(self, delta):
        '''Given a delta point, compares this weight of the current pickpoint
        to determine if it's worth indexing the point based on certain
        heuristics.
        
        This is done randomly, and the chance that a re-index
        happens depends on the weight of the point. The weightier the point,
        the less likely a reindex will happen (since each change to the point
        is less significant. Weightier deltas increase the likelihood.

        The tolerance is the maximum weight at which a point is guaranteed to
        be re - indexed (assuming the delta weight is 1). E.g. if the tolerance
        is 10, then the point will be re - indexed until it's weight is 10. When
        the weight reaches 20, reindexing happens 50% of the time. At 100, 10%
        of the time. At 1000, 1% of the time. Etc.
        
        The pp_min_weight and pp_min_predictability can weed out inaccurate
        points.
        
        Returns -1 if the point should be UNindexed
        Returns 0 if the indexed point can remain unchanged
        Returns 1 if the point should be reindexed.
        '''
        if (not self.pickpoint_indexable() or
            self.pickpoint.weight < self.pp_reindex_min_weight or
            self.pickpoint.predictability < self.pp_reindex_min_predictability):
            return (-1)
        try:
            chance = (self.pp_reindex_tolerance *
                      (delta.weight / self.pickpoint.weight))
        except ZeroDivisionError:
            chance = 1.0
        if chance >= 1.0 or random.random() < chance:
            return 1
        else:
            return 0


class PickPointHash(db.Model):
    '''Stores one (of many) PickPoint hashes for the PickSpace position
    of some entity.
    The key_name is '%s/%s' % (entity.kind() , entity.key().id_or_name()).
    '''
    entity_kind = db.StringProperty(required=True)
    predictability = db.FloatProperty(required=True, default=0.0)
    hashes = db.ListProperty(int, required=True)

    @classmethod
    def make_key_name(cls, entity_key):
        return '%s/%s' % (entity_key.kind(), entity_key.id_or_name())

    @classmethod
    def gen_keys(cls, entity_keys):
        '''Given a list of PickspaceModel entities, returns the PickPointHash
        keys for them.'''
        for key in entity_keys:
            yield db.Key.from_path(cls.kind(), cls.make_key_name(key),
                                   parent=key)

    @classmethod
    def batch_get(cls, entity_keys):
        '''Given a list of PickspaceModel entities, generates PickPointHash
        objects for them, either from the db, or newly created. If update
        is True, the hash will be updated with the entities pickpoint.'''
        for key, hash in izip(entity_keys, db.get(cls.gen_keys(entity_keys))):
            if hash is None:
                hash = cls(key_name=cls.make_key_name(key),
                           parent=key, entity_kind=key.kind())
            yield hash

    @classmethod
    def batch_update(cls, data):
        '''Given a list of tuples containing a entity key and a pickpoint, 
        updates the hashed index for that entity.'''
        toput = []
        points = list(datum[1] for datum in data)
        hashes = list(cls.batch_get(list(datum[0] for datum in data)))
        for hash, point in izip(hashes, points):
            hash.hashes = point.get_slshes()
            hash.predictability = point.predictability
            toput.append(hash)
        db.put(toput)

    @classmethod
    def batch_delete(cls, entity_keys):
        '''Given a list of entity keys, deletes any hashes stored for them.'''
        db.delete(list(cls.gen_keys(entity_keys)))

    MAX_SUBQUERIES = 5
    BUFFER_SIZE = 20

    @classmethod
    def search(cls, point, kinds=None, limit=None):
        '''Given a point, yields keys with similar points, in very approximate
        order of similarity. This generator may return any number of results,
        including none. It is up to the caller to sort the results more
        precisely.
        kinds - a list of model kinds. None for all kinds.
        limit - don't yield more than this many results.'''
        if isinstance(kinds, list):
            hashes_per_query = max(cls.MAX_SUBQUERIES / len(kinds), 1)
        else:
            hashes_per_query = cls.MAX_SUBQUERIES
        i = 0
        previous_matches = set()
        result_buffer = list()
        hashes = point.get_slshes()
        while hashes:
            search_hashes = hashes[:hashes_per_query]
            hashes = hashes[hashes_per_query:]
            query = cls.all()
            if kinds:
                if isinstance(kinds, list):
                    query.filter('entity_kind IN', kinds)
                else:
                    query.filter('entity_kind =', kinds)
            query.filter('hashes IN', search_hashes)
            for result in query:
                entity_key = result.parent_key()
                if entity_key in previous_matches:
                    continue
                result_buffer.append(entity_key)
                if len(result_buffer) >= cls.BUFFER_SIZE:
                    for entity in db.get(result_buffer):
                        if entity:
                            yield entity
                    result_buffer = list()
                previous_matches.add(entity_key)
                i += 1
                if limit and limit <= i:
                    break
            if limit and limit <= i:
                break
        # Clear the result buffer
        for entity in db.get(result_buffer):
            if entity:
                yield entity


'''
from pickspace import PickPoint
from pickspace.db import PickPointHash
from model.user import get_user_by_username

mary = get_user_by_username('mary')
p = PickPoint(None, 1, 0)
q = PickPoint(None, 0.001, 0)
PickPointHash.reindex(mary.key(), p)
p = p + q
PickPointHash.reindex(mary.key(), p)
'''


''' test:
from pickspace.db import *

name = 'test%d' % random.randint(0,1000)
points = []
for i in xrange(20):
    point = PickPoint()
    point.weight = random.random() * 10
    point.predictability = random.random()
    points.append(point)
    PickPointShard.increment(name, point, 5)
    
total = PickPointShard.get_point_by_name(name)
rtotal = PickPoint.sum(points) 
print 'Total: ', total
print 'Real total: ', rtotal
print 'Similarity: ', total.similarity(rtotal)
print 'sdistance: ', total.sdistance(rtotal)'''
