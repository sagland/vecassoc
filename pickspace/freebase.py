'''
Testing ground for pickspace associations using freebase data.
'''

import os, random, csv
from pickspace.pickpoint2 import PickPoint2
from gdata.analytics import Property

DATADIR = '/home/sagland/data/freebase'

INFLUENCE_PULL = 0.75
INFLUENCEE_PULL = 0.25
PEER_PULL = 0.5

def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def read_file(file):
    filepath = os.path.join(DATADIR, file) + '.tsv'
    reader = unicode_csv_reader(open(filepath, 'rb'), delimiter='\t',
                                quoting=csv.QUOTE_NONE, escapechar='\\')
    for line in reader:
        yield line

class Entity(object):

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __cmp__(self, other):
        return cmp(self.name, other.name)

    def __ne__(self, other):
        return self.name != other.name

    def __init__(self, name):

        self.name = name
        self.influences = set()
        self.influencees = set()
        self.peers = set()
        self.point = PickPoint2(weight=1.0) # Random position with a small weight

    @property
    def all_influences(self):
        try:
            return self._all_influences
        except AttributeError:
            self._all_influences = set()
            for influence in self.influences:
                if influence not in self._all_influences and self is not influence:
                    self._all_influences.add(influence)
                    self._all_influences.update(influence.all_influences)
        return self._all_influences

    @property
    def all_influencees(self):
        try:
            return self._all_influencees
        except AttributeError:
            self._all_influencees = set()
            for influencee in self.influencees:
                if influencee not in self._all_influencees and self is not influencee:
                    self._all_influencees.add(influencee)

        return self._all_influencees

    @property
    def nrelations(self):
        return len(self.influences) + len(self.influencees) + len(self.peers)

    @property
    def influence(self):
        return len(self.all_influencees)

    def generate_gravity(self):
        '''Generates pickpoints representing the pull of associated entities.'''
        for influence in self.influences:
            yield influence.point * INFLUENCE_PULL
        for influencee in self.influencees:
            yield influencee.point * INFLUENCEE_PULL
        for peer in self.peers:
            yield peer.point * PEER_PULL

    @property
    def gravity(self):
        '''Returns a pickpoint representing the total "pull" on this entity
        by related entities.'''
        return PickPoint2.sum(self.generate_gravity(), predictor=self.point)

    def store_gravity(self):
        '''Collect and remember the current cumulative "pull".'''
        self._gravity = self.gravity

    def apply_gravity(self):
        self.point += self._gravity ** 0.5

    def iterate_gravity(self):
        '''Shifts the entities point in the direction the summed pull of
        related entities.'''
        self.store_gravity()
        self.apply_gravity()

    @staticmethod
    def cmpinfluence(a, b):
        return cmp(a.influence, b.influence)

    def point_cmp(self, e1, e2):
        '''Ranks other entities based on their point similarity to this one.'''
        return self.point.cmp(e1.point, e2.point)

class InfluenceSphere(object):
    '''Analysis of influence between influential people.'''

    INFLUENCE_FILE = 'influence/influence_node'
    PEER_FILE = 'influence/peer_relationship'


    def __init__(self):

        self._entities = {}

        influence_file = read_file(self.INFLUENCE_FILE)
        influence_file.next()
        for influence in influence_file:
            name, id, influences, influencees, peers = influence
            if not name or (not influences and not influencees):
                continue
            entity = self.entity(name)
            for influence in (self.entity(_) for _ in self.listsplit(influences)):
                entity.influences.add(influence)
                influence.influencees.add(entity)
            for influencee in (self.entity(_) for _ in self.listsplit(influencees)):
                entity.influencees.add(influencee)
                influencee.influences.add(entity)

        peer_file = read_file(self.PEER_FILE)
        peer_file.next()
        for line in peer_file:
            peers = set(self.entity(_) for _ in self.listsplit(line[2]))
            for peer in peers:
                peer.peers.update(peers - set([peer]))


    def blend_align(self):
        alignments = list()
        peerpairs = set()
        for entity in self.entities:
            for influence in entity.influences:
                alignments.append((influence, entity, 0.25, 0.75))
            for peer in entity.peers:
                if set([peer, entity]) not in peerpairs:
                    alignments.append((peer, entity, 1.0, 1.0))

        opc = -1
        na = len(alignments)
        ITERATIONS = 10
        for i in range(ITERATIONS):
            random.shuffle(alignments)
            for j, (e1, e2, w1, w2) in enumerate(alignments):
                r1, r2 = PickPoint2.attract(e1.point, e2.point, w1, w2)
                if r1:
                    e1.point = e1.point + r1
                if r2:
                    e2.point = e2.point + r2
                pc = int(100.0 * ((i * na) + j) / float(ITERATIONS * na))
                if pc > opc:
                    print 'Aligning: %d%%' % pc
                    opc = pc

    def align(self):
        '''New attempt at alignment using summing rather than progressive
        blending.'''

        opc = -1
        ne = self.nentities
        ITERATIONS = 3
        for i in range(ITERATIONS):
            for j, entity in enumerate(self.entities):
                entity.store_gravity()
                pc = int(100.0 * ((i * ne * 2) + j) / float(ITERATIONS * ne * 2))
                if pc > opc:
                    print 'Aligning: %d%%' % pc
                    opc = pc
            for j, entity in enumerate(self.entities):
                entity.apply_gravity()
                pc = int(100.0 * ((i * ne * 2) + j + ne) / float(ITERATIONS * ne * 2))
                if pc > opc:
                    print 'Aligning: %d%%' % pc
                    opc = pc

        print 'Average similarity:', self.avg_similarity(self.entities, 10000)
        #platoclub = self.entity('Plato').all_influencees
        #print 'Plato similarity:', self.avg_similarity(platoclub, 1000)
        #beatles = self.entity('The Beatles')
        #beatlesclub = beatles.all_influencees.union(beatles.all_influences)
        #print 'Beatles similarity:', self.avg_similarity(beatlesclub, 1000)
        #for artist in beatlesclub:
        #    print artist

        subject = self.entity('Charlie Chaplin')
        for i, e in enumerate(self.neighbours(subject, 1)):
            print (i + 1, e, '%.2d%%' % (subject.point.similarity(e.point) * 100),
                   e.point.weight)
            if i >= 1000:
                break



    def neighbours(self, entity, min_weight=0):
        '''Yields the "nearest neighbours" to an entity.'''
        for e in sorted(self.entities, entity.point_cmp, reverse=True):
            if e != entity and e.point.weight > min_weight:
                yield e

    @staticmethod
    def avg_similarity(entities, ntests=1000):
        entities = list(entities)
        avg = 0.0
        for i in range(ntests):
            e1 = random.choice(entities)
            while True:
                e2 = random.choice(entities)
                if e1 != e2:
                    break
            avg += e1.point.similarity_fast(e2.point)
        return avg / ntests

    @property
    def entities(self):
        return self._entities.itervalues()

    @property
    def nentities(self):
        return len(self._entities)

    @staticmethod
    def listsplit(list):
        return [_.replace('^^', ', ') for _ in
                list.replace(', ', '^^').split(',') if _]

    def entity(self, name):
        try:
            return self._entities[name]
        except KeyError:
            return self._entities.setdefault(name, Entity(name))



if __name__ == '__main__':
    influence_sphere = InfluenceSphere()
    influence_sphere.align()



